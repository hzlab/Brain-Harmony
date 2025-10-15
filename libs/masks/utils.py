# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks, cls_token=False):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    if cls_token:
        cls_t = x[:, :1, :]
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [
                torch.cat(
                    (cls_t, torch.gather(x[:, 1:, :], dim=1, index=mask_keep)), dim=1
                )
            ]
    else:
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)
