import glob
import gzip
import json
import math
import os
import pickle
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from absl import logging
from torch.utils.data import Dataset
from tqdm import tqdm


def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example:
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ("Wrong dimension! dim=%d." % nd)
    return data_crop


class fmri_BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        raise NotImplementedError("please implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("please implement __getitem__")

    def generate_masking(self, signal_length, max_length=490):
        mask = torch.zeros(max_length, dtype=torch.int)
        mask[:signal_length] = 1
        return mask

    def load_statistical_param_file(self, statistic_dir, norm_type="robust"):
        """
        Determine the filename for normalization parameters based on the current mode,
        normalization type, and state of the class.

        The filename and subsequent computation differ if self.preprocess is True.

        Parameters:
            statistic_dir (str): Directory to store/load parameter files.
            norm_type (str): Either 'all_robust_scale' (for median/IQR) or 'all_mean_std' (for mean/std).
                            (Note: In direct mode, only robust parameters are computed.)
        """
        os.makedirs(statistic_dir, exist_ok=True)
        prefix = "meanstd_" if norm_type == "all_mean_std" else ""
        postfix = f"_{self.preprocess}" if self.preprocess else ""
        if hasattr(self, "normparam_name_postfix") and self.normparam_name_postfix:
            postfix = postfix + "_" + self.normparam_name_postfix
        if self.use_train_statistic or self.state == "pretrain":
            size = 450 if self.use_subcortical else 400
            filename = (
                f"{self.dataset_name}_NormParams_Pretrain_{prefix}{size}{postfix}.npz"
            )
        elif self.state == "downstream":
            suffix = "" if self.use_subcortical else "_400"
            split_id = self.split.split("_")[-1]
            filename = f"{self.dataset_name}_NormParams_Downstream_{split_id}_{self.label_name}_{prefix}{suffix}{postfix}.npz"
        else:
            raise NotImplementedError(f"Unsupported state: {self.state}")

        self.params_file = os.path.join(statistic_dir, filename)
        print(f"Loading parameters from file {self.params_file}")

    def _load_or_compute_normalization_params(self, norm_type="robust"):
        """
        Load the normalization parameters from file if available; otherwise,
        compute and save them.

        In direct mode (self.preprocess is True), the function always returns robust
        parameters computed from the directly standardized data.

        Parameters:
            norm_type (str): Either 'robust' or 'meanstd'. (Ignored in direct mode.)

        Returns:
            dict: A dictionary of normalization parameters:
                For robust: {'medians': ..., 'iqrs': ...}
                For meanstd: {'mean': ..., 'std': ...}
        """
        if os.path.exists(self.params_file):
            params = np.load(self.params_file)
            if norm_type == "all_mean_std":
                mean = params["mean"]
                std = params["std"]
                logging.info(
                    f"Normalization parameters loaded from file: {self.params_file}"
                )
                return {"mean": mean, "std": std}
            else:
                medians = params["medians"]
                iqrs = params["iqrs"]
                logging.info(
                    f"Normalization parameters loaded from file: {self.params_file}"
                )
                return {"medians": medians, "iqrs": iqrs}
        else:
            if self.state == "downstream" and self.use_train_statistic:
                raise NotImplementedError(
                    f"Computation for training statistics is not implemented params file: {self.params_file}."
                )
            else:
                return self._compute_normalization_params(norm_type)

    def _compute_normalization_params(self, norm_type="robust"):
        """
        Compute normalization parameters over all subjects.

        When self.preprocess is True (direct mode), each subject's data is first standardized
        (i.e. zero-mean and unit-variance per channel) and then robust parameters (median and IQR)
        are computed across subjects and channels.

        In standard mode, the data are used as-is and per-subject means are computed, from which
        either robust (median/IQR) or mean/std parameters are calculated.

        Parameters:
            norm_type (str): Either 'robust' or 'meanstd'. (Ignored in direct mode.)

        Returns:
            dict: A dictionary of computed normalization parameters.
        """
        print(f"Computing normalization parameters and saving to {self.params_file}.")
        all_data = []
        all_data_mean = []

        for idx in tqdm(range(self.__len__())):
            data = self.load_signal(idx)

            if self.preprocess == "centering":
                data = data - np.mean(data, axis=1)[:, None]
                all_data.append(data)
            elif self.preprocess == "zscore":
                data = (data - np.mean(data, axis=1)[:, None]) / (
                    np.std(data, axis=1)[:, None] + 1e-9
                )
                all_data.append(data)
            else:
                all_data.append(data)
                temp_data = np.mean(data, axis=1)
                all_data_mean.append(temp_data)

        if self.preprocess:
            all_data = np.stack(all_data)
            if norm_type == "all_mean_std":
                mean = np.mean(all_data, axis=(0, 2))
                std = np.std(all_data, axis=(0, 2))
                np.savez(self.params_file, mean=mean, std=std)
                return {"mean": mean, "std": std}
            else:
                medians = np.median(all_data, axis=(0, 2))
                iqrs = np.percentile(all_data, 75, axis=(0, 2)) - np.percentile(
                    all_data, 25, axis=(0, 2)
                )
                np.savez(self.params_file, medians=medians, iqrs=iqrs)
                return {"medians": medians, "iqrs": iqrs}
        else:
            try:
                all_data = np.stack(all_data)
            except Exception:
                pass
            all_data_mean = np.stack(all_data_mean)
            if norm_type == "all_mean_std":
                mean = np.mean(all_data_mean, axis=0)
                std = np.std(all_data_mean, axis=0)
                np.savez(self.params_file, mean=mean, std=std)
                return {"mean": mean, "std": std}
            else:
                medians = np.median(all_data_mean, axis=0)
                iqrs = np.percentile(all_data_mean, 75, axis=0) - np.percentile(
                    all_data_mean, 25, axis=0
                )
                np.savez(self.params_file, medians=medians, iqrs=iqrs)
                return {"medians": medians, "iqrs": iqrs}

    def _get_start_end_idx(self, fmri_size, clip_size):
        """
        Sample a clip of size clip_size from a video of size video_size and
        return the indices of the first and last frame of the clip. If clip_idx is
        -1, the clip is randomly sampled, otherwise uniformly split the video to
        num_clips clips, and select the start and end index of clip_idx-th video
        clip.
        Args:
            video_size (int): number of overall frames.
            clip_size (int): size of the clip to sample from the frames.
            clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
                clip_idx is larger than -1, uniformly split the video to num_clips
                clips, and select the start and end index of the clip_idx-th video
                clip.
            num_clips (int): overall number of clips to uniformly sample from the
                given video for testing.
        Returns:
            start_idx (int): the start frame index.
            end_idx (int): the end frame index.
        """
        delta = max(fmri_size - clip_size, 0)
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx

    def _temporal_sampling(self, frames, start_idx, end_idx, num_samples):
        """
        Given the start and end frame index, sample num_samples frames between
        the start and end with equal interval.
        Args:
            frames (tensor): a tensor of video frames, dimension is
                `num video frames` x `channel` x `height` x `width`.
            start_idx (int): the index of the start frame.
            end_idx (int): the index of the end frame.
            num_samples (int): number of frames to sample.
        Returns:
            frames (tersor): a tensor of temporal sampled video frames, dimension is
                `num clip frames` x `channel` x `height` x `width`.
        """
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        new_frames = torch.index_select(frames, 1, index)
        return new_frames

    def preprocess_fmri(self, ts_array):
        if self.preprocess == "centering":
            ts_array = ts_array - np.mean(ts_array, axis=1)[:, None]
        elif self.preprocess == "zscore":
            ts_array = (ts_array - np.mean(ts_array, axis=1)[:, None]) / (
                np.std(ts_array, axis=1)[:, None] + 1e-9
            )

        # Apply robust scaling
        if self.norm == "all_robust_scaling":
            median, iqr = (
                self.normalization_params["medians"],
                self.normalization_params["iqrs"],
            )
            ts_array = (ts_array - median[:, None]) / iqr[:, None]
        elif self.norm == "all_mean_std":
            mean, std = (
                self.normalization_params["mean"],
                self.normalization_params["std"],
            )
            ts_array = (ts_array - mean[:, None]) / std[:, None]
        else:
            pass 

        return ts_array

    def pad(self, ts_array, original_time_length):
        padded = torch.zeros((400, self.target_pad_length), dtype=ts_array.dtype)
        assert original_time_length <= self.target_pad_length
        padded[:, :original_time_length] = ts_array[:, :original_time_length]

        return padded

    def signal_attn_mask(self):
        if self.target_num_patches is None:
            arange = torch.arange(self.num_patches)
            attn_mask_ = (
                arange[None, :] < self.num_patches
            ).int() 
            mask2d = attn_mask_.repeat(400, 1)
            return mask2d.view(-1)
        arange = torch.arange(self.target_num_patches)
        attn_mask_ = (arange[None, :] < self.num_patches).int() 
        mask2d = attn_mask_.repeat(400, 1)
        return mask2d.view(-1)


class T1_BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        raise NotImplementedError("please implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("please implement __getitem__")

    def load_mri_image(self, path):
        """Load a NIfTI image and return its data array."""
        img_data = torch.load(path, weights_only=True)
        return img_data.numpy()

    def normalize(self, img):
        img = (img - img.mean()) / (img.std() + 1e-9)
        return crop_center(
            torch.tensor(img.copy(), dtype=torch.float32), (160, 192, 160)
        )


########################################################################################################################
#                                                   UKB fMRI dataset                                                   #
########################################################################################################################
class UKBDataset(fmri_BaseDataset):
    def __init__(
        self,
        state="pretrain",
        split="train",
        n_cortical_rois=400,
        n_subcortical_rois=50,
        seq_length=490,
        statistic_dir="",
        fmri_data_dir="",
        split_ids_file="",
        use_subcortical=True,
        norm="all_robust_scaling",
        preprocess=None,
        downsample=False,
        sampling_rate=1,
        label_name="",
        labels_file="",
        label_norm_stat_file="",
        label_normalization=False,
        use_train_statistic=False,
        return_length=False,
        normparam_name_postfix="",
        standard_time=48 * 0.735,
        target_num_patches=None,
        if_fusion=False,
        **kwargs,
    ):
        self.state = state
        self.split = split
        self.label_norm_stat_file = label_norm_stat_file

        if use_subcortical:
            self.n_rois = n_cortical_rois + n_subcortical_rois
        else:
            self.n_rois = n_cortical_rois
        self.seq_length = seq_length

        self.use_subcortical = use_subcortical
        self.cortical_file = f"fMRI.Schaefer17n{n_cortical_rois}p.csv.gz"
        if use_subcortical:
            self.subcortical_file = f"fMRI.Tian_Subcortex_{self._map_subcortical(n_subcortical_rois)}_3T.csv.gz"

        self.fmri_data_dir = fmri_data_dir
        self.use_train_statistic = use_train_statistic
        self.return_length = return_length
        self.normparam_name_postfix = normparam_name_postfix

        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.num_frames = int(seq_length // sampling_rate)
        if downsample:
            self.target_tr = 0.735 * sampling_rate
        else:
            self.target_tr = 0.735

        self.patch_size = round(standard_time / self.target_tr)
        self.num_patches = math.ceil(self.seq_length // sampling_rate / self.patch_size)
        if target_num_patches is not None:
            assert target_num_patches >= self.num_patches
            self.target_pad_length = target_num_patches * self.patch_size
        else:
            self.target_pad_length = (
                self.num_patches * self.patch_size
            )
        self.target_num_patches = target_num_patches
        print(
            f"ukb seq length: {self.seq_length // sampling_rate} target pad length: {self.target_pad_length}, num_patches: {self.num_patches} num_patches after pad: {self.target_num_patches}"
        )

        with open(split_ids_file, "rb") as f:
            train_val_test_ids = pickle.load(f)
        if split == "val":
            self.ids = (
                train_val_test_ids["val_train_ids"]
                + train_val_test_ids["val_val_ids"]
            )
        elif split == "all":
            self.ids = (
                train_val_test_ids["train_ids"]
                + train_val_test_ids["val_train_ids"]
                + train_val_test_ids["val_val_ids"]
                + train_val_test_ids["val_test_ids"]
            )
        else:
            self.ids = train_val_test_ids[f"{split}_ids"]

        if if_fusion:
            base_dir = Path("/path/to/T1_data")
            folder_names = [p.name for p in base_dir.iterdir() if p.is_dir()]

            unique_id_list = [name for name in self.ids if name in folder_names]
            self.ids = unique_id_list

        self.preprocess = preprocess
        self.dataset_name = "ukb"

        self.norm = norm
        if norm:
            self.load_statistical_param_file(statistic_dir, norm_type=norm)
            self.normalization_params = self._load_or_compute_normalization_params(
                norm_type=norm
            )

    def __len__(self):
        return len(self.ids)

    def load_signal(self, idx):
        id = self.ids[idx]
        ts_cortical = self._load_ts(id, self.cortical_file)
        if self.use_subcortical:
            ts_subcortical = self._load_ts(id, self.subcortical_file)
            ts_array = np.concatenate((ts_subcortical, ts_cortical), axis=0).astype(
                np.float32
            )
        else:
            ts_array = ts_cortical.astype(np.float32)
        assert ts_array.shape == (self.n_rois, self.seq_length)

        return ts_array

    def load_fmri(self, idx):
        ts_array = self.load_signal(idx)
        ts_array = self.preprocess_fmri(ts_array)

        if self.downsample:
            clip_size = self.sampling_rate * self.num_frames
            start_idx, end_idx = self._get_start_end_idx(self.seq_length, clip_size)
            ts_array = self._temporal_sampling(
                torch.from_numpy(ts_array), start_idx, end_idx, self.num_frames
            )
        else:
            ts_array = torch.from_numpy(ts_array)
        original_time_length = ts_array.shape[1]
        padded = self.pad(ts_array, original_time_length)
        ts = torch.unsqueeze(padded, 0).to(torch.float32)

        return ts, original_time_length

    def __getitem__(self, idx):
        ts, original_time_length = self.load_fmri(idx)
        if self.state == "downstream":
            label = self.load_fmri_label(idx)
            if self.return_length:
                return ts, label, 490
            return ts, label
        if self.return_length:
            attn_mask = self.signal_attn_mask()
            return ts, original_time_length, self.patch_size, attn_mask
        return ts

    def _map_subcortical(self, n_subcortical_rois):
        mapping = {16: "S1", 32: "S2", 50: "S3", 54: "S4"}
        return mapping[n_subcortical_rois]

    def save_normalization_params(
        self, medians, iqrs, filename="normalization_params_train.csv"
    ):
        df = pd.DataFrame(
            {"roi_index": range(len(medians)), "median": medians, "iqr": iqrs}
        )
        df.to_csv(self.params_file, index=False)
        logging.info(f"Normalization parameters saved to {self.params_file}")

    def _load_csv(self, id, file):
        file_path = os.path.join(self.fmri_data_dir, id, file)
        return pd.read_csv(file_path)

    def _load_ts(self, id, file):
        df = self._load_csv(id, file)
        return df.iloc[:, 1 : self.seq_length + 1].values

    def _load_roi_names(self):
        df_cortical = self._load_csv(self.ids[0], self.cortical_file)
        df_subcortical = self._load_csv(self.ids[0], self.subcortical_file)
        cortical_names = df_cortical["label_name"].values
        subcortical_names = df_subcortical["label_name"].values
        return np.concatenate((subcortical_names, cortical_names), axis=0)


########################################################################################################################
#                                                   UKB T1 dataset                                                     #
########################################################################################################################
class UKB_T1_Dataset(T1_BaseDataset):
    def __init__(
        self,
        state,
        T1_data_dir="",
        split="train",
        split_ids_file="",
        label_name="",
        labels_file="",
        label_normalization=False,
        if_fusion=False,
        **kwargs,
    ):
        self.state = state
        self.T1_data_dir = T1_data_dir
        self.label_normalization = label_normalization

        with open(split_ids_file, "rb") as f:
            train_val_test_ids = pickle.load(f)
        if split == "val":
            self.ids = (
                train_val_test_ids["val_train_ids"]
                + train_val_test_ids["val_val_ids"]
            )
        elif split == "all":
            self.ids = (
                train_val_test_ids["train_ids"]
                + train_val_test_ids["val_train_ids"]
                + train_val_test_ids["val_val_ids"]
                + train_val_test_ids["val_test_ids"]
            )
        else:
            self.ids = train_val_test_ids[f"{split}_ids"]

        if if_fusion:
            base_dir = Path("/path/to/T1_data")
            folder_names = [p.name for p in base_dir.iterdir() if p.is_dir()]

            unique_id_list = [name for name in self.ids if name in folder_names]
            self.ids = unique_id_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img = self.load_t1(index)
        return img

    def load_t1(self, index):
        file_name = os.path.join(
            self.ids[index], "cropped/images", self.ids[index] + ".pt"
        )
        img_path = os.path.join(self.T1_data_dir, file_name)
        img = self.load_mri_image(img_path)

        pad_width = [(6, 9), (2, 4), (0, 22)]
        img = np.pad(img, pad_width, mode="constant", constant_values=0)

        img_tensor = self.normalize(img)
        img_tensor = img_tensor.to(torch.float32).unsqueeze(0)

        return img_tensor


########################################################################################################################
#                                                   UKB fusion dataset                                                 #
########################################################################################################################
class UKB_FusionDataset(UKBDataset, UKB_T1_Dataset):
    def __init__(self, **kwargs):
        UKBDataset.__init__(self, **kwargs)
        UKB_T1_Dataset.__init__(self, **kwargs)
        logging.info("UKB FusionDataset initialized")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ts, _ = self.load_fmri(index)
        attn_mask = self.signal_attn_mask()

        t1 = self.load_t1(index)
        return ts, t1, self.patch_size, attn_mask, self.ids[index]


########################################################################################################################
#                                                   ABIDE I fMRI dataset                                               #
########################################################################################################################


class ABIDE_I_fMRI_Dataset(fmri_BaseDataset):
    def __init__(
        self,
        split="train",
        n_cortical_rois=400,
        n_subcortical_rois=50,
        use_subcortical=True,
        fmri_data_dir="ABIDE1_fMRI_par",
        splits_file="data_splits.json",
        tr_file="TR.json",  
        tr_min=None,  # for select data
        tr_max=None,  # for select data
        standard_time=48 * 0.735,
        target_num_patches=10,
        return_ori_length=False,
        **kwargs,
    ):
        self.n_cortical_rois = n_cortical_rois
        if use_subcortical:
            self.n_subcortical_rois = n_subcortical_rois
            self.n_rois = n_cortical_rois + n_subcortical_rois
        else:
            self.n_rois = n_cortical_rois

        self.use_subcortical = use_subcortical
        self.fmri_data_dir = fmri_data_dir
        self.split = split
        self.return_ori_length = return_ori_length

        self.tr_min = tr_min
        self.tr_max = tr_max
        self.target_num_patches = target_num_patches

        # Load data splits
        with open(splits_file, "r") as f:
            splits = json.load(f)

        if split == "all":
            self.data = splits["train"] + splits["val"] + splits["test"]
        else:
            self.data = splits[split]

        with open(tr_file, "r") as f:
            self.tr_dict = json.load(f)

        self.target_tr = 0.01
        # based on tr min and tr max, select site used for a specific tr
        if (self.tr_min is not None) and (self.tr_max is not None):
            filtered = []
            for item in self.data:
                site = item["id"].split("/")[0]
                tr_val = self.tr_dict.get(site)
                if (tr_val is not None) and (self.tr_min <= tr_val < self.tr_max):
                    filtered.append(item)
                    self.target_tr = tr_val
            self.data = filtered
            print(
                f"Filtered by TR in [{self.tr_min}, {self.tr_max}): {len(self.data)} samples remain target tr: {self.target_tr}"
            )

            self.patch_size = round(standard_time / self.target_tr)

        else:
            NotImplementedError

        # Define file patterns
        self.cortical_file = "Schaefer17n%sp.csv.gz" % n_cortical_rois
        if use_subcortical:
            self.subcortical_file = "Tian_Subcortex_S3_3T.csv.gz"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_data, label, patch_size, attn_mask, original_time_length, subject_id = (
            self.load_fmri(idx)
        )

        if self.return_ori_length:
            return tensor_data, label, patch_size, attn_mask, original_time_length
        return tensor_data, label, patch_size, attn_mask

    def load_fmri(self, idx):
        item = self.data[idx]
        subject_id = item["id"]
        subject_number = subject_id.split("/")[-1].split("-")[-1]
        
        cortical_path = os.path.join(
            self.fmri_data_dir, subject_number, f"{subject_number}_{self.cortical_file}"
        )
        ts_cortical = self._load_csv(cortical_path)

        if self.use_subcortical:
            subcortical_path = os.path.join(
                self.fmri_data_dir,
                subject_number,
                f"{subject_number}_{self.subcortical_file}",
            )
            ts_subcortical = self._load_csv(subcortical_path)
            ts_array = np.concatenate((ts_subcortical, ts_cortical), axis=0).astype(
                np.float32
            )
        else:
            ts_array = ts_cortical.astype(np.float32)

        original_time_length = ts_array.shape[1]

        self.num_patches = math.ceil(original_time_length / self.patch_size)
        if self.target_num_patches is not None:
            assert self.target_num_patches >= self.num_patches
            self.target_pad_length = self.target_num_patches * self.patch_size
        else:
            self.target_pad_length = self.num_patches * self.patch_size

        padded = self.pad(torch.from_numpy(ts_array), original_time_length)
        ts = torch.unsqueeze(padded, 0).to(torch.float32)

        label = torch.tensor(item["label"] - 1, dtype=torch.long)  # Convert 1/2 to 0/1

        attn_mask = self.signal_attn_mask()
        return ts, label, self.patch_size, attn_mask, original_time_length, subject_id

    def _load_csv(self, filepath):
        with gzip.open(filepath, "rt") as f:
            df = pd.read_csv(f)
            ts = df.iloc[:, 1:].values
        return ts


########################################################################################################################
#                                                   ABIDE I T1 dataset                                                 #
########################################################################################################################
class ABIDE_T1_Dataset(T1_BaseDataset):
    def __init__(self, T1_data_dir, splits_file, split="train", **kwargs):
        self.T1_data_dir = T1_data_dir

        # Load split information
        with open(splits_file, "r") as f:
            splits = json.load(f)

        if split == "all":
            self.t1_samples = splits["train"] + splits["val"] + splits["test"]
        else:
            self.t1_samples = splits[split]

    def __len__(self):
        return len(self.t1_samples)

    def load_mri_image(self, path):
        img_nifti = nib.load(path)
        img_data = img_nifti.get_fdata()
        return img_data

    def __getitem__(self, index):
        return self.load_t1(index)

    def load_t1(self, index, fmri_samples=None):
        if fmri_samples is not None:
            sample = fmri_samples[index]
        else:
            sample = self.t1_samples[index]

        subject_id = sample["id"].split("-")[-1]
        if not os.path.exists(
            os.path.join(self.T1_data_dir, f"{subject_id}_T1.nii.gz")
        ):
            subject_id = sample["id"].split("-")[-1][2:]
        file_name = f"{subject_id}_T1.nii.gz"
        img_path = os.path.join(self.T1_data_dir, file_name)

        img = self.load_mri_image(img_path)

        pad_width = [(6, 9), (2, 4), (0, 22)]
        img = np.pad(img, pad_width, mode="constant", constant_values=0)
        img_tensor = self.normalize(img)  # test 是否做 scaling
        img_tensor = img_tensor.to(torch.float32).unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.long) - 1
        return img_tensor, label


########################################################################################################################
#                                                   ABIDE I Fusion dataset                                             #
########################################################################################################################


class ABIDE_I_FusionDataset(ABIDE_I_fMRI_Dataset, ABIDE_T1_Dataset):
    def __init__(self, **kwargs):
        ABIDE_I_fMRI_Dataset.__init__(self, **kwargs)
        ABIDE_T1_Dataset.__init__(self, **kwargs)
        logging.info("AbideI FusionDataset initialized")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ts, label, self.patch_size, attn_mask, original_time_length, subject_id = (
            self.load_fmri(index)
        )

        t1, _ = self.load_t1(index, fmri_samples=self.data)

        return (
            ts,
            t1,
            label,
            self.patch_size,
            attn_mask,
            original_time_length,
            subject_id,
        )


########################################################################################################################
#                                                   load embed dataset                                                   #
########################################################################################################################
class GenerateEmbedDataset(Dataset):
    def __init__(self, root_dir, portion=1.0, seed=42):
        self.root_dir = root_dir
        self.portion = portion

        pattern = os.path.join(root_dir, "*.npz")
        all_files = sorted(glob.glob(pattern))

        self.files = all_files

        if len(all_files) == 0:
            raise RuntimeError(f"No .npz files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        arr = np.load(filepath)
        tensor = torch.from_numpy(arr["data"])
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor.squeeze(), arr["attn_mask"].squeeze()


class GenerateEmbedDataset_downstream(Dataset):
    def __init__(self, root_dir, splits_file, split, portion=1.0, seed=42):
        self.root_dir = root_dir
        if split != "train":
            portion = 1.0
        self.portion = portion

        with open(splits_file, "r") as f:
            splits = json.load(f)

        if split == "all":
            all_samples = splits["train"] + splits["val"] + splits["test"]
        else:
            all_samples = splits[split]
            
        self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        subject_name = item["id"]
        filepath = os.path.join(self.root_dir, f"{subject_name.replace('/', '~')}.npz")
        arr = np.load(filepath)
        tensor = torch.from_numpy(arr["data"])
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        return tensor.squeeze(), np.squeeze(arr["label"]), arr["attn_mask"].squeeze()


def get_dataset(name, **kwargs):
    if name == "UKB_fusion":
        return UKB_FusionDataset(**kwargs)
    elif name == "fMRI":
        return UKBDataset(**kwargs)
    else:
        raise NotImplementedError(name)


def get_downstream_dataset(name, **kwargs):
    if name == "AbideI":
        dataset = ABIDE_I_fMRI_Dataset(**kwargs)

    else:
        raise Exception(f"this {name} dataset not used in downstream task")
    return dataset


def get_downstream_fusion_dataset(name, **kwargs):
    if name == "AbideI":
        dataset = ABIDE_I_FusionDataset(**kwargs)
    else:
        raise Exception(f"this {name} dataset not used in downstream task")
    return dataset
