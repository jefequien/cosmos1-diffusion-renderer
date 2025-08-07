# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from megatron.core import parallel_state
from tqdm import tqdm

from cosmos_predict1.diffusion.conditioner import VideoDiffusionRendererCondition
from cosmos_predict1.diffusion.model.model_t2w import DiffusionT2WModel, broadcast_condition, split_inputs_cp, cat_outputs_cp, EDMEulerScheduler


class DiffusionRendererModel(DiffusionT2WModel):
    def __init__(self, config):
        super().__init__(config)

        # Custom configs
        self.condition_keys = config.condition_keys
        self.condition_drop_rate = config.condition_drop_rate
        self.append_condition_mask = config.append_condition_mask

        # Update scheduler
        self.scheduler = EDMEulerScheduler(sigma_max=80, sigma_min=0.02, sigma_data=self.sigma_data)

    def prepare_diffusion_renderer_latent_conditions(
            self, data_batch: dict[str, Tensor],
            condition_keys: list[str] = ["rgb"], condition_drop_rate: float = 0, append_condition_mask: bool = True,
            dtype: torch.dtype = None, device: torch.device = None,
            latent_shape: Union[Tuple[int, int, int, int, int], torch.Size] = None,
            mode="train",
    ) -> Tensor:
        if latent_shape is None:
            B, C, T, H, W = data_batch[condition_keys[0]].shape
            latent_shape = (B, 16, T // 8 + 1, H // 8, W // 8)
        if append_condition_mask:
            latent_mask_shape = (latent_shape[0], 1, latent_shape[2], latent_shape[3], latent_shape[4])
        if dtype is None:
            dtype = data_batch[condition_keys[0]].dtype
        if device is None:
            device = data_batch[condition_keys[0]].device

        latent_condition_list = []
        for cond_key in condition_keys:
            ## Note: relaxed this constraint so during training the model can also train with missing attributes.
            # if cond_key not in data_batch and mode == "train":
            #     raise KeyError(f"Condition key '{cond_key}' is missing in data_batch during 'train' mode. "
            #                                f"Expected keys: {condition_keys}")
            is_condition_dropped = condition_drop_rate > 0 and np.random.rand() < condition_drop_rate
            is_condition_skipped = cond_key not in data_batch
            if is_condition_dropped or is_condition_skipped:
                # Dropped or skipped condition
                condition_state = torch.zeros(latent_shape, dtype=dtype, device=device)
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    condition_mask = torch.zeros(latent_mask_shape, dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)
            else:
                # Valid condition
                condition_state = data_batch[cond_key].to(device=device, dtype=dtype)
                condition_state = self.encode(condition_state).contiguous()
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    condition_mask = torch.ones(latent_mask_shape, dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)

        return torch.cat(latent_condition_list, dim=1)

    def _get_conditions(
        self,
        data_batch: Dict,
        is_negative_prompt: bool = False,
    ):
        # Latent state
        raw_state = data_batch[self.input_data_key]
        with torch.no_grad():
            latent_condition = self.prepare_diffusion_renderer_latent_conditions(
                data_batch,
                condition_keys=self.condition_keys,
                condition_drop_rate=0,
                append_condition_mask=self.append_condition_mask,
                dtype=raw_state.dtype, device=raw_state.device, latent_shape=None, mode="inference",
            )

        data_batch["latent_condition"] = latent_condition
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        return condition, uncondition

    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 0.0,
        seed: int = 1000,
        state_shape: tuple | None = None,
        n_sample: int | None = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 15,
    ) -> Tensor:
        """Generate samples from a data batch using diffusion sampling.

        This function generates samples from either image or video data batches using diffusion sampling.
        It handles both conditional and unconditional generation with classifier-free guidance.

        Args:
            data_batch (dict): Raw data batch from the training data loader
            guidance (float, optional): Classifier-free guidance weight. Defaults to 1.5.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            state_shape (tuple | None, optional): Shape of the state tensor. Uses self.state_shape if None. Defaults to None.
            n_sample (int | None, optional): Number of samples to generate. Defaults to 1.
            is_negative_prompt (bool, optional): Whether to use negative prompt for unconditional generation. Defaults to False.
            num_steps (int, optional): Number of diffusion sampling steps. Defaults to 35.

        Returns:
            Tensor: Generated samples after diffusion sampling
        """
        # num_steps = 4
        self.scheduler.set_timesteps(num_steps)

        xt = torch.randn(size=(n_sample, 2) + tuple(state_shape)) * self.scheduler.init_noise_sigma
        to_cp = self.net.is_context_parallel_enabled
        if to_cp:
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)

        rgb = data_batch["rgb"]
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            xt = xt.to(**self.tensor_kwargs)
            xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
            # Predict the noise residual
            t = t.to(**self.tensor_kwargs)
            # net_output_cond = self.net(x=xt_scaled, timesteps=t, **condition.to_dict())
            # net_output = net_output_cond
            # if guidance > 0:
            #     net_output_uncond = self.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
            #     net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
            # Compute the previous noisy sample x_t -> x_t-1
            # xt = self.scheduler.step(net_output, t, xt).prev_sample

            # latent_starts = torch.arange((state_shape[1] // 2)) * 2
            latent_starts = torch.arange((state_shape[1] // 4)) * 4
            print(latent_starts)
            
            xt_output = torch.zeros_like(xt)
            print(xt.shape)

            # for latent_starts_chunk in latent_starts.reshape((-1, 8)):
            #     print(latent_starts_chunk)
            #     rgb_list = []
            #     for latent_start in latent_starts_chunk:
            #         rgb_list.extend([rgb[:,:,8 * latent_start + 7:8 * latent_start + 8, ...]] * 8)
            #     data_batch["rgb"] = torch.cat(rgb_list, dim=2)
            #     data_batch["rgb"] = data_batch["rgb"][:,:,7:,...] # Drop first 7 frames
            #     condition, _ = self._get_conditions(data_batch, is_negative_prompt)
            #     x = xt_scaled[:,0,:,latent_starts_chunk,:,:]
            #     brick_output = self.net(x=x, timesteps=t, **condition.to_dict())
            #     xt_output[:,0,:, latent_starts_chunk, :,:] += brick_output * 0.5


            for latent_start in tqdm(latent_starts):
                latent_end = latent_start + 8
                rgb_start = 32 * (latent_start // 4)
                rgb_end = 32 * (latent_end // 4)
                # print(latent_start, latent_end, rgb_start, rgb_end)
                
                if rgb_end <= rgb.shape[2]:
                    data_batch["rgb"] = rgb[:,:,rgb_start: rgb_end, ...]
                else:
                    data_batch["rgb"] = torch.cat([rgb[:,:,rgb_start:, ...], rgb[:,:,: rgb_end - rgb.shape[2], ...]], dim=2)
                data_batch["rgb"] = data_batch["rgb"][:,:,7:,...] # Drop first 7 frames
                condition, _ = self._get_conditions(data_batch, is_negative_prompt)

                x0 = xt_scaled[:,0,:, latent_start: latent_start+2, ...]
                if latent_end <= xt_scaled.shape[3]:
                    x1 = xt_scaled[:,1,:, latent_start+2: latent_end, ...]
                else:
                    r = latent_end - xt_scaled.shape[3]
                    x1 = torch.cat([
                        xt_scaled[:,1,:,latent_start+2:, ...], 
                        xt_scaled[:,1,:,:r, ...]
                    ], dim=2)
                x = torch.cat([x0, x1], dim=2)
                brick_output = self.net(x=x, timesteps=t, **condition.to_dict())

                if latent_end <= xt_scaled.shape[3]:
                    # xt_output[:,0,:, latent_start:latent_start+1, :,:] += brick_output[:,:,0:1,...]
                    # xt_output[:,0,:, latent_start+1:latent_end:2, :,:] += (brick_output[:,:,1:8:2,...] / 4.0)
                    # xt_output[:,0,:, latent_start+1:latent_end:2, :,:] += (brick_output[:,:,1:8:2,...] / 4.0)
                    # xt_output[:,1,:, latent_start+2:latent_end, :,:] += (brick_output[:,:,2:8,...] / 3.0)
                    xt_output[:,0,:, latent_start+0:latent_start+1, :,:] += brick_output[:,:,0:1,...]
                    xt_output[:,0,:, latent_start+1:latent_start+4, :,:] += (brick_output[:,:,1:4,...] / 2.0)
                    xt_output[:,0,:, latent_start+5:latent_start+8, :,:] += (brick_output[:,:,5:8,...] / 2.0)
                    xt_output[:,1,:, latent_start+2:latent_start+4, :,:] += (brick_output[:,:,2:4,...] / 2.0)
                    xt_output[:,1,:, latent_start+4:latent_start+6, :,:] += brick_output[:,:,4:6,...]
                    xt_output[:,1,:, latent_start+6:latent_start+8, :,:] += (brick_output[:,:,6:8,...] / 2.0)
                else:
                    # l = xt_scaled.shape[3] - latent_start
                    # r = latent_end - xt_scaled.shape[3]
                    # xt_output[:,0,:, latent_start:latent_start+1, :,:] += brick_output[:,:,0:1,...]
                    # xt_output[:,0,:, latent_start+1:latent_start+l:2, :,:] += (brick_output[:,:,1:l:2,...] / 4.0)
                    # xt_output[:,0,:, 1:r:2, :,:] += (brick_output[:,:,l+1::2,...] / 4.0)
                    # xt_output[:,1,:, latent_start+2: , :,:] += (brick_output[:,:,2:l,...] / 3.0)
                    # xt_output[:,1,:, :r , :,:] += (brick_output[:,:,l:,...] / 3.0)
                    xt_output[:,0,:, latent_start+0:latent_start+1, :,:] += brick_output[:,:,0:1,...]
                    xt_output[:,0,:, latent_start+1:latent_start+4, :,:] += (brick_output[:,:,1:4,...] / 2.0)
                    xt_output[:,0,:, 1:4, :,:] += (brick_output[:,:,5:8,...] / 2.0)
                    xt_output[:,1,:, latent_start+2:latent_start+4, :,:] += (brick_output[:,:,2:4,...] / 2.0)
                    xt_output[:,1,:, 0:2, :,:] += brick_output[:,:,4:6,...]
                    xt_output[:,1,:, 2:4, :,:] += (brick_output[:,:,6:8,...] / 2.0)
                
            xt = self.scheduler.step(xt_output, t, xt).prev_sample
        # samples = xt

        samples_list = []
        # latent_starts = torch.arange((state_shape[1] // 2)) * 2
        latent_starts = torch.arange((state_shape[1] // 4)) * 4
        for latent_start in latent_starts:
            latent_end = latent_start + 8
            sample0 = xt[:,0,:, latent_start: latent_start+2, ...]
            # sample0 = xt[:,1,:, latent_start: latent_start+2, ...]
            if latent_end <= xt.shape[3]:
                sample1 = xt[:,1,:,latent_start+2:latent_end,:,:]
            else:
                sample1 = torch.cat([
                    xt[:,1,:,latent_start+2:, ...], 
                    xt[:,1,:,: latent_end - xt.shape[3], ...],
                ], dim=2)
            sample = torch.cat([sample0, sample1], dim=2)
            samples_list.append(sample)
        samples = torch.cat(samples_list, dim=2)

        if to_cp:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

# def float_slice_along_D(latent, start_d, end_d, out_d):
#     """
#     Extract float-aligned depth slice from `latent` (B, C, D, H, W).
#     Returns (B, C, out_d, H, W).
#     """
#     B, C, D, H, W = latent.shape
#     device = latent.device
#     dtype = latent.dtype  # Match dtype

#     # Depth coordinates from start_d to end_d
#     d_coords = torch.linspace(start_d, end_d, steps=out_d, device=device, dtype=dtype)
#     d_coords = d_coords / (D - 1) * 2 - 1  # normalize to [-1, 1]

#     # Grid coords
#     grid_d = d_coords.view(out_d, 1, 1).expand(out_d, H, W)
#     grid_h = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, H, 1).expand(out_d, H, W)
#     grid_w = torch.linspace(-1, 1, W, device=device, dtype=dtype).view(1, 1, W).expand(out_d, H, W)

#     grid = torch.stack((grid_d, grid_h, grid_w), dim=-1).unsqueeze(0)  # (1, out_d, H, W, 3)

#     # Sample
#     sliced = F.grid_sample(
#         latent, grid, mode='bilinear', padding_mode='border', align_corners=True
#     )
#     return sliced

# def apply_subpixel_patch_along_D(latent, update_patch, latent_start, latent_end):
#     """
#     Applies update_patch (1, C, d, H, W) into latent (1, C, D, H, W),
#     interpolated over float range [latent_start, latent_end] in D.
#     """
#     assert latent.dim() == 5 and update_patch.dim() == 5
#     B, C, D, H, W = latent.shape
#     _, _, d, h, w = update_patch.shape
#     assert B == 1, "Only batch size = 1 is supported"

#     device = latent.device
#     dtype = latent.dtype

#     # Create target depth grid for patch
#     d_coords = torch.linspace(latent_start, latent_end, steps=d, device=device, dtype=dtype)
#     d_coords_norm = d_coords / (D - 1) * 2 - 1  # Normalize to [-1, 1]

#     # Build full sampling grid
#     grid_d = d_coords_norm.view(d, 1, 1).expand(d, h, w)
#     grid_h = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(1, h, 1).expand(d, h, w)
#     grid_w = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, 1, w).expand(d, h, w)
#     grid = torch.stack((grid_d, grid_h, grid_w), dim=-1).unsqueeze(0)  # (1, d, h, w, 3)

#     # Warp update_patch into latent's coordinate frame
#     warped_patch = F.grid_sample(
#         update_patch, grid, mode='bilinear', padding_mode='zeros', align_corners=True
#     )
#     warped_mask = F.grid_sample(
#         torch.ones_like(update_patch), grid, mode='bilinear', padding_mode='zeros', align_corners=True
#     )

#     # Determine integer slice range
#     d_start_idx = max(int(torch.floor(latent_start)), 0)
#     d_end_idx = min(int(torch.ceil(latent_end)), D)

#     # Figure out the mapping range for slicing
#     latent_slice_len = d_end_idx - d_start_idx
#     if latent_slice_len != d:
#         # Resize warped_patch/mask to match latent slice length
#         warped_patch = F.interpolate(warped_patch, size=(latent_slice_len, h, w), mode='trilinear', align_corners=True)
#         warped_mask = F.interpolate(warped_mask, size=(latent_slice_len, h, w), mode='trilinear', align_corners=True)

#     # Insert into latent
#     latent_slice = latent[:, :, d_start_idx:d_end_idx, :, :]
#     blended_slice = latent_slice * (1 - warped_mask) + warped_patch
#     latent[:, :, d_start_idx:d_end_idx, :, :] = blended_slice

#     return latent
