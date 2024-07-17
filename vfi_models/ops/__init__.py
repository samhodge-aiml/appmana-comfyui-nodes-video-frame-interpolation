import torch.multiprocessing as mp

from ...config.add_configuration import VideoFrameInterpolationConfiguration

if mp.current_process().name == "MainProcess":
    from comfy.cli_args import args  # type: VideoFrameInterpolationConfiguration

    ops_backend = args.vfi_ops_backend or "cupy"

    assert ops_backend in ["taichi", "cupy"]

    if ops_backend == "taichi":
        from .taichi_ops import softsplat, ModuleSoftsplat, FunctionSoftsplat, softsplat_func, costvol_func, sepconv_func, init, batch_edt, FunctionAdaCoF, ModuleCorrelation, FunctionCorrelation, _FunctionCorrelation
    else:
        from .cupy_ops import softsplat, ModuleSoftsplat, FunctionSoftsplat, softsplat_func, costvol_func, sepconv_func, init, batch_edt, FunctionAdaCoF, ModuleCorrelation, FunctionCorrelation, _FunctionCorrelation
