from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch import Tensor


class PatchEmbed(nn.Module):

    def __init__(self, img_size=(450, 490), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0]) * (img_size[1] // patch_size)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = (img_size[0], img_size[1] // patch_size)

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 32,
        in_chans: int = 1,
        embed_dim: int = 768,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size)
        )

        self.interpolation = interpolation
        self.antialias = antialias


        self.pinvs = {}

    def _cache_pinvs(self) -> dict:
        pinvs = {}
        for ps in self.patch_size_seq:
            ps = (1, ps)
            pinvs[ps] = self._calculate_pinv((1, self.patch_size), ps)
        return pinvs

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, old_shape: Tuple[int, int], new_shape: Tuple[int, int]
    ) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: Tuple[int, int]):
        if self.patch_size == new_patch_size:
            return patch_embed

        self.pinvs[new_patch_size] = self._calculate_pinv(
            (1, int(self.patch_size)), new_patch_size
        )
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = torch.vmap(
            torch.vmap(resample_patch_embed, 0, 0), 1, 1
        )

        return v_resample_patch_embed(patch_embed)

    def forward(
        self,
        x: Tensor,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_patch_size: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:

        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, (1, patch_size))

        x = F.conv2d(x, weight, bias=self.proj.bias, stride=(1, patch_size))

        x = x.flatten(2).transpose(1, 2)


        if return_patch_size:
            return x, patch_size

        return x
