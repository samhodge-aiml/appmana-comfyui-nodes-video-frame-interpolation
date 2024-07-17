import gc
import typing

import einops
import requests
import torch
from can_ada import parse, URL

from comfy.cmd.folder_paths import add_model_folder_path, supported_pt_extensions
from comfy.model_downloader import get_or_download, add_known_models
from comfy.model_downloader_types import UrlFile
from comfy.model_management import soft_empty_cache, get_torch_device

VFI_MODELS_FOLDER_NAME: typing.Final[str] = "vfi_models"
_base_model_urls = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
]


def _get_github_release_file_urls(owner, repo, tag):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    response = requests.get(url)

    if response.status_code == 200:
        release_data = response.json()
        assets = release_data['assets']

        return [asset['browser_download_url'] for asset in assets]
    else:
        return []


def _initialize_known_models():
    add_model_folder_path(VFI_MODELS_FOLDER_NAME, extensions=supported_pt_extensions | {".onnx"})

    for url in _base_model_urls:
        parsed_url: URL = parse(url)
        path_pieces = [piece for piece in parsed_url.pathname.split("/") if piece != ""]
        owner = path_pieces[0]
        repo = path_pieces[1]
        tag = path_pieces[-1]

        file_urls = _get_github_release_file_urls(owner, repo, tag)
        url_files = [UrlFile(file_url) for file_url in file_urls]
        add_known_models(VFI_MODELS_FOLDER_NAME, *url_files)


class InterpolationStateList():

    def __init__(self, frame_indices: typing.List[int], is_skip_list: bool):
        self.frame_indices = frame_indices
        self.is_skip_list = is_skip_list

    def is_frame_skipped(self, frame_index):
        is_frame_in_list = frame_index in self.frame_indices
        return self.is_skip_list and is_frame_in_list or not self.is_skip_list and not is_frame_in_list


class MakeInterpolationStateList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_indices": ("STRING", {"multiline": True, "default": "1,2,3"}),
                "is_skip_list": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = ("INTERPOLATION_STATES",)
    FUNCTION = "create_options"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def create_options(self, frame_indices: str, is_skip_list: bool):
        frame_indices_list = [int(item) for item in frame_indices.split(',')]

        interpolation_state_list = InterpolationStateList(
            frame_indices=frame_indices_list,
            is_skip_list=is_skip_list,
        )
        return (interpolation_state_list,)


def load_file_from_github_release(model_type, ckpt_name):
    return get_or_download(VFI_MODELS_FOLDER_NAME, ckpt_name)


def load_file_from_direct_url(model_type, url):
    url_file = UrlFile(url)
    return get_or_download(VFI_MODELS_FOLDER_NAME, url_file.filename, [url_file])


def preprocess_frames(frames):
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")


def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()


def assert_batch_size(frames, batch_size=2, vfi_name=None):
    subject_verb = "Most VFI models require" if vfi_name is None else f"VFI model {vfi_name} requires"
    assert len(
        frames) >= batch_size, f"{subject_verb} at least {batch_size} frames to work with, only found {frames.shape[0]}. Please check the frame input using PreviewImage."


def _generic_frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.Union[typing.SupportsInt, typing.List],
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: InterpolationStateList = None,
        use_timestep=True,
        dtype=torch.float16,
        final_logging=True):
    # https://github.com/hzwer/Practical-RIFE/blob/main/inference_video.py#L169
    def non_timestep_inference(frame0, frame1, n):
        middle = return_middle_frame_function(frame0, frame1, None, *return_middle_frame_function_args)
        if n == 1:
            return [middle]
        first_half = non_timestep_inference(frame0, middle, n=n // 2)
        second_half = non_timestep_inference(middle, frame1, n=n // 2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    output_frames = torch.zeros(multiplier * frames.shape[0], *frames.shape[1:], dtype=dtype, device="cpu")
    out_len = 0

    number_of_frames_processed_since_last_cleared_cuda_cache = 0

    for frame_itr in range(len(frames) - 1):  # Skip the final frame since there are no frames after it
        frame0 = frames[frame_itr:frame_itr + 1]
        output_frames[out_len] = frame0  # Start with first frame
        out_len += 1
        # Ensure that input frames are in fp32 - the same dtype as model
        frame0 = frame0.to(dtype=torch.float32)
        frame1 = frames[frame_itr + 1:frame_itr + 2].to(dtype=torch.float32)

        if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
            continue

        # Generate and append a batch of middle frames
        middle_frame_batches = []

        if use_timestep:
            for middle_i in range(1, multiplier):
                timestep = middle_i / multiplier

                middle_frame = return_middle_frame_function(
                    frame0.to(get_torch_device()),
                    frame1.to(get_torch_device()),
                    timestep,
                    *return_middle_frame_function_args
                ).detach().cpu()
                middle_frame_batches.append(middle_frame.to(dtype=dtype))
        else:
            middle_frames = non_timestep_inference(frame0.to(get_torch_device()), frame1.to(get_torch_device()),
                                                   multiplier - 1)
            middle_frame_batches.extend(torch.cat(middle_frames, dim=0).detach().cpu().to(dtype=dtype))

        # Copy middle frames to output
        for middle_frame in middle_frame_batches:
            output_frames[out_len] = middle_frame
            out_len += 1

        number_of_frames_processed_since_last_cleared_cuda_cache += 1
        # Try to avoid a memory overflow by clearing cuda cache regularly
        if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
            print("Comfy-VFI: Clearing cache...", end=' ')
            soft_empty_cache()
            number_of_frames_processed_since_last_cleared_cuda_cache = 0
            print("Done cache clearing")

        gc.collect()

    if final_logging:
        print(f"Comfy-VFI done! {len(output_frames)} frames generated at resolution: {output_frames[0].shape}")
    # Append final frame
    output_frames[out_len] = frames[-1:]
    out_len += 1
    # clear cache for courtesy
    if final_logging:
        print("Comfy-VFI: Final clearing cache...", end=' ')
    soft_empty_cache()
    if final_logging:
        print("Done cache clearing")
    return output_frames[:out_len]


def generic_frame_loop(
        model_name,
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.Union[typing.SupportsInt, typing.List],
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: InterpolationStateList = None,
        use_timestep=True,
        dtype=torch.float32):
    assert_batch_size(frames, vfi_name=model_name.replace('_', ' ').replace('VFI', ''))
    if type(multiplier) == int:
        return _generic_frame_loop(
            frames,
            clear_cache_after_n_frames,
            multiplier,
            return_middle_frame_function,
            *return_middle_frame_function_args,
            interpolation_states=interpolation_states,
            use_timestep=use_timestep,
            dtype=dtype
        )
    if type(multiplier) == list:
        multipliers = list(map(int, multiplier))
        multipliers += [2] * (len(frames) - len(multipliers) - 1)
        frame_batches = []
        for frame_itr in range(len(frames) - 1):
            multiplier = multipliers[frame_itr]
            if multiplier == 0: continue
            frame_batch = _generic_frame_loop(
                frames[frame_itr:frame_itr + 2],
                clear_cache_after_n_frames,
                multiplier,
                return_middle_frame_function,
                *return_middle_frame_function_args,
                interpolation_states=interpolation_states,
                use_timestep=use_timestep,
                dtype=dtype,
                final_logging=False
            )
            if frame_itr != len(frames) - 2:  # Not append last frame unless this batch is the last one
                frame_batch = frame_batch[:-1]
            frame_batches.append(frame_batch)
        output_frames = torch.cat(frame_batches)
        print(f"Comfy-VFI done! {len(output_frames)} frames generated at resolution: {output_frames[0].shape}")
        return output_frames
    raise NotImplementedError(f"multipiler of {type(multiplier)}")


class FloatToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0, 'min': 0, 'step': 0.01})
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "ComfyUI-Frame-Interpolation"

    def convert(self, float):
        if hasattr(float, "__iter__"):
            return (list(map(int, float)),)
        return (int(float),)


""" def generic_4frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.SupportsInt,
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: InterpolationStateList = None,
        use_timestep=False):
    
    if use_timestep: raise NotImplementedError("Timestep 4 frame VFI model")
    def non_timestep_inference(frame_0, frame_1, frame_2, frame_3, n):        
        middle = return_middle_frame_function(frame_0, frame_1, None, *return_middle_frame_function_args)
        if n == 1:
            return [middle]
        first_half = non_timestep_inference(frame_0, middle, n=n//2)
        second_half = non_timestep_inference(middle, frame_1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half] """

_initialize_known_models()
