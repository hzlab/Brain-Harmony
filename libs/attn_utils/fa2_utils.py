import importlib.metadata
from functools import lru_cache

import torch
from packaging import version


def is_flash_attn_2_available():
    # if not is_torch_available():
    #     return False

    # if not _is_package_available("flash_attn"):
    #     return False

    # # Let's add an extra check to see if cuda is available
    # import torch

    # if not (torch.cuda.is_available() or is_torch_mlu_available()):
    #     return False

    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(
            "2.1.0"
        )
    # elif torch.version.hip:
    #     # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
    #     return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    # elif is_torch_mlu_available():
    #     return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.3.3")
    else:
        return False


@lru_cache()
def is_flash_attn_greater_or_equal(library_version: str):
    # if not _is_package_available("flash_attn"):
    #     return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(
        library_version
    )


@lru_cache()
def is_flash_attn_greater_or_equal_2_10():
    # if not _is_package_available("flash_attn"):
    #     return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(
        "2.1.0"
    )
