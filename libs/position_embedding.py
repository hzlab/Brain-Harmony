import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class PosEmbed_Base(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.config = config
        grid_size = config.grid_size
        self.grid_size = grid_size
        embed_dim = config.embed_dim

        self.device = device

        self.repeat_time = config.grid_size[1]

        self.grid = self.get_grid(grid_size)

        self.emb_h_encoder = nn.Parameter(
            torch.zeros(grid_size[0] * grid_size[1], embed_dim // 2),
            requires_grad=False,
        )

        pos_emb_h_encoder = self.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, self.grid[0]
        )  # (H*W, D/2)

        self.emb_h_encoder.data.copy_(torch.from_numpy(pos_emb_h_encoder).float())

        self.use_pos_embed_decoder = False
        if config.use_pos_embed_decoder:
            predictor_embed_dim = config.predictor_embed_dim
            self.use_pos_embed_decoder = True
            self.emb_h_decoder = nn.Parameter(
                torch.zeros(grid_size[0] * grid_size[1], predictor_embed_dim // 2),
                requires_grad=False,
            )
            pos_emb_h_decoder = self.get_1d_sincos_pos_embed_from_grid(
                predictor_embed_dim // 2, self.grid[0]
            )  # (H*W, D/2)
            self.emb_h_decoder.data.copy_(torch.from_numpy(pos_emb_h_decoder).float())

    def get_grid(self, grid_size):
        grid_h = np.arange(grid_size[0], dtype=float)
        grid_w = np.arange(grid_size[1], dtype=float)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
        return grid

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb


class SineCosine_PosEmbed(PosEmbed_Base):
    def __init__(self, device, config):
        super().__init__(device, config)

        grid_size = config.grid_size
        embed_dim = config.embed_dim
        self.cls_token = config.cls_token

        self.emb_w_encoder = nn.Parameter(
            torch.zeros(grid_size[0] * grid_size[1], embed_dim // 2),
            requires_grad=False,
        )
        pos_emb_w_encoder = self.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, self.grid[1]
        )  # (H*W, D/2)
        self.emb_w_encoder.data.copy_(torch.from_numpy(pos_emb_w_encoder).float())

        if config.use_pos_embed_decoder:
            predictor_embed_dim = config.predictor_embed_dim
            self.emb_w_decoder = nn.Parameter(
                torch.zeros(grid_size[0] * grid_size[1], predictor_embed_dim // 2),
                requires_grad=False,
            )
            pos_emb_w_decoder = self.get_1d_sincos_pos_embed_from_grid(
                predictor_embed_dim // 2, self.grid[1]
            )  # (H*W, D/2)
            self.emb_w_decoder.data.copy_(torch.from_numpy(pos_emb_w_decoder).float())

    def forward(self):
        emb_encoder = torch.cat(
            [self.emb_h_encoder, self.emb_w_encoder], dim=1
        ).unsqueeze(0)  # (H*W, D)
        if self.use_pos_embed_decoder:
            emb_decoder = torch.cat(
                [self.emb_h_decoder, self.emb_w_decoder], dim=1
            ).unsqueeze(0)  # (H*W, D)

        if self.cls_token:
            pos_embed_encoder = torch.concat(
                [
                    torch.zeros([1, 1, emb_encoder.shape[2]]).to(self.device),
                    emb_encoder,
                ],
                dim=1,
            )
            if self.use_pos_embed_decoder:
                pos_embed_decoder = torch.concat(
                    [
                        torch.zeros([1, 1, emb_decoder.shape[2]]).to(self.device),
                        emb_decoder,
                    ],
                    dim=1,
                )
        else:
            pos_embed_encoder = emb_encoder
            if self.use_pos_embed_decoder:
                pos_embed_decoder = emb_decoder
        if self.use_pos_embed_decoder:
            return pos_embed_encoder, pos_embed_decoder
        return pos_embed_encoder, None


class BrainGradient_GeometricHarmonics_Anatomical_400_PosEmbed(PosEmbed_Base):
    def __init__(self, device, config):
        super().__init__(device, config)

        embed_dim = config.embed_dim
        self.cls_token = config.cls_token
        geoh_dim = config.geoh_dim
        geo_harm = config.geo_harm
        grad_dim = config.grad_dim
        gradient = config.gradient

        self.geo_harm_proj = nn.Linear(geoh_dim, embed_dim // 2)
        self.grad_proj = nn.Linear(grad_dim, embed_dim // 2)

        df = pd.read_csv(geo_harm, header=None)
        self.geo_harm = torch.tensor(
            df.values[1:, 1:].astype(np.float32), dtype=torch.float32
        ).to(device, non_blocking=True)

        df = pd.read_csv(gradient, header=None)
        self.gradient = torch.tensor(df.values, dtype=torch.float32).to(
            device, non_blocking=True
        )

        if config.use_pos_embed_decoder:
            self.decoder_pos_embed_proj = nn.Linear(
                config.embed_dim // 2, config.predictor_embed_dim // 2
            )

    def forward(self):
        geo_harm_pos_embed = self.geo_harm_proj(self.geo_harm)
        gradient_pos_embed = self.grad_proj(self.gradient)

        pos_embed = (gradient_pos_embed + geo_harm_pos_embed) * 0.5
        emb_w = pos_embed.squeeze().repeat_interleave(self.repeat_time, dim=0)
        emb_w = (emb_w - emb_w.min()) / (emb_w.max() - emb_w.min()) * 2 - 1

        emb_encoder = torch.cat([self.emb_h_encoder, emb_w], dim=1).unsqueeze(
            0
        )  # (H*W, D)

        if self.cls_token:
            pos_embed_encoder = torch.concat(
                [
                    torch.zeros([1, 1, emb_encoder.shape[2]], requires_grad=False).to(
                        self.device
                    ),
                    emb_encoder,
                ],
                dim=1,
            )
        else:
            pos_embed_encoder = emb_encoder

        if self.use_pos_embed_decoder:
            emb_w_decoder = self.decoder_pos_embed_proj(emb_w.detach())
            emb_decoder = torch.cat(
                [self.emb_h_decoder, emb_w_decoder], dim=1
            ).unsqueeze(0)  # (H*W, D)
            if self.cls_token:
                pos_embed_decoder = torch.concat(
                    [
                        torch.zeros(
                            [1, 1, emb_decoder.shape[2]], requires_grad=False
                        ).to(self.device),
                        emb_decoder,
                    ],
                    dim=1,
                )
            else:
                pos_embed_decoder = emb_decoder
            return pos_embed_encoder, pos_embed_decoder
        return pos_embed_encoder, None


class SineCosine3D_PosEmbed(PosEmbed_Base):
    def __init__(self, device, config):
        super().__init__(device, config)

        grid_size = config.grid_size  # Should be [H, W, D]
        embed_dim = config.embed_dim
        self.cls_token = config.cls_token

        # Create position embeddings for each dimension
        self.emb_h_encoder = nn.Parameter(
            torch.zeros(grid_size[0] * grid_size[1] * grid_size[2], embed_dim // 3),
            requires_grad=False,
        )
        self.emb_w_encoder = nn.Parameter(
            torch.zeros(grid_size[0] * grid_size[1] * grid_size[2], embed_dim // 3),
            requires_grad=False,
        )
        self.emb_d_encoder = nn.Parameter(
            torch.zeros(grid_size[0] * grid_size[1] * grid_size[2], embed_dim // 3),
            requires_grad=False,
        )

        # Get 3D grid
        grid = self.get_3d_grid(grid_size)

        # Generate position embeddings for each dimension
        pos_emb_h_encoder = self.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 3, grid[0]
        )  # (H*W*D, D/3)
        pos_emb_w_encoder = self.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 3, grid[1]
        )  # (H*W*D, D/3)
        pos_emb_d_encoder = self.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 3, grid[2]
        )  # (H*W*D, D/3)

        # Copy position embeddings to parameters
        self.emb_h_encoder.data.copy_(torch.from_numpy(pos_emb_h_encoder).float())
        self.emb_w_encoder.data.copy_(torch.from_numpy(pos_emb_w_encoder).float())
        self.emb_d_encoder.data.copy_(torch.from_numpy(pos_emb_d_encoder).float())

        # For decoder if needed
        if config.use_pos_embed_decoder:
            predictor_embed_dim = config.predictor_embed_dim
            self.use_pos_embed_decoder = True

            self.emb_h_decoder = nn.Parameter(
                torch.zeros(
                    grid_size[0] * grid_size[1] * grid_size[2], predictor_embed_dim // 3
                ),
                requires_grad=False,
            )
            self.emb_w_decoder = nn.Parameter(
                torch.zeros(
                    grid_size[0] * grid_size[1] * grid_size[2], predictor_embed_dim // 3
                ),
                requires_grad=False,
            )
            self.emb_d_decoder = nn.Parameter(
                torch.zeros(
                    grid_size[0] * grid_size[1] * grid_size[2], predictor_embed_dim // 3
                ),
                requires_grad=False,
            )

            pos_emb_h_decoder = self.get_1d_sincos_pos_embed_from_grid(
                predictor_embed_dim // 3, grid[0]
            )
            pos_emb_w_decoder = self.get_1d_sincos_pos_embed_from_grid(
                predictor_embed_dim // 3, grid[1]
            )
            pos_emb_d_decoder = self.get_1d_sincos_pos_embed_from_grid(
                predictor_embed_dim // 3, grid[2]
            )

            self.emb_h_decoder.data.copy_(torch.from_numpy(pos_emb_h_decoder).float())
            self.emb_w_decoder.data.copy_(torch.from_numpy(pos_emb_w_decoder).float())
            self.emb_d_decoder.data.copy_(torch.from_numpy(pos_emb_d_decoder).float())

    def get_3d_grid(self, grid_size):
        """Generate a 3D grid."""
        grid_h = np.arange(grid_size[0], dtype=float)
        grid_w = np.arange(grid_size[1], dtype=float)
        grid_d = np.arange(grid_size[2], dtype=float)

        grid = np.meshgrid(grid_w, grid_h, grid_d)  # w, h, d order
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([3, -1])  # [3, H*W*D]
        return grid

    def forward(self):
        # Concatenate embeddings for encoder
        emb_encoder = torch.cat(
            [self.emb_h_encoder, self.emb_w_encoder, self.emb_d_encoder], dim=1
        ).unsqueeze(0)  # (1, H*W*D, D)

        # Add cls token if needed
        if self.cls_token:
            pos_embed_encoder = torch.concat(
                [
                    torch.zeros([1, 1, emb_encoder.shape[2]]).to(self.device),
                    emb_encoder,
                ],
                dim=1,
            )
        else:
            pos_embed_encoder = emb_encoder

        # Handle decoder embeddings if needed
        if self.use_pos_embed_decoder:
            emb_decoder = torch.cat(
                [self.emb_h_decoder, self.emb_w_decoder, self.emb_d_decoder], dim=1
            ).unsqueeze(0)
            if self.cls_token:
                pos_embed_decoder = torch.concat(
                    [
                        torch.zeros([1, 1, emb_decoder.shape[2]]).to(self.device),
                        emb_decoder,
                    ],
                    dim=1,
                )
            else:
                pos_embed_decoder = emb_decoder
            return pos_embed_encoder, pos_embed_decoder

        return pos_embed_encoder, None
