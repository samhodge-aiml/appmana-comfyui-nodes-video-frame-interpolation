from typing import Optional, Literal

import configargparse

from comfy.cli_args_types import Configuration


def add_configuration(parser: configargparse.ArgParser) -> configargparse.ArgParser:
    parser.add_argument("--vfi-ops-backend",
                        type=str,
                        help=f"The operations backend. One of cupy or taichi",
                        env_var="VFI_OPS_BACKEND")
    return parser


class VideoFrameInterpolationConfiguration(Configuration):
    def __init__(self):
        super().__init__()
        self.vfi_ops_backend: Literal["taichi", "cupy"] = "cupy"
