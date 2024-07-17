import logging

from comfy.cli_args import args  # type: VideoFrameInterpolationConfiguration
from ...config.add_configuration import VideoFrameInterpolationConfiguration

ops_backend = args.vfi_ops_backend or "cupy"
assert ops_backend in ["taichi", "cupy"]

try:
    if ops_backend == "taichi":
        from .taichi_ops import softsplat, ModuleSoftsplat, FunctionSoftsplat, softsplat_func, costvol_func, sepconv_func, init, batch_edt, FunctionAdaCoF, ModuleCorrelation, FunctionCorrelation, _FunctionCorrelation
    else:
        try:
            from .cupy_ops import softsplat, ModuleSoftsplat, FunctionSoftsplat, softsplat_func, costvol_func, sepconv_func, init, batch_edt, FunctionAdaCoF, ModuleCorrelation, FunctionCorrelation, _FunctionCorrelation
        except ImportError:
            from .taichi_ops import softsplat, ModuleSoftsplat, FunctionSoftsplat, softsplat_func, costvol_func, sepconv_func, init, batch_edt, FunctionAdaCoF, ModuleCorrelation, FunctionCorrelation, _FunctionCorrelation
except Exception as exc_info:
    logging.warning("error importing ops", exc_info=exc_info)
