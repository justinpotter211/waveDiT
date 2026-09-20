import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from waveform_scaling import (
    encode_relative_geometry,
    haversine_km,
    normalize_waveform_array,
    project_condition_value,
    uses_epicentral_distance,
    uses_relative_geometry,
)


EXPECTED_WAVEFORM_CHANNELS = 3


class LoadWaveform(Dataset):
    def __init__(
        self,
        wf_dir,
        sta_lat_min,
        sta_lat_max,
        sta_lon_min,
        sta_lon_max,
        src_lat_min,
        src_lat_max,
        src_lon_min,
        src_lon_max,
        dep_min,
        dep_max,
        mag_min,
        mag_max,
        min_get,
        max_get,
        target_length,
        waveform_mean_value,
        waveform_std_value,
        abs_geo_encoder_type="none",
        epi_dist_min=-100.0,
        epi_dist_max=2500.0,
    ):
        self.wf_dir = wf_dir
        self.wf_path = sorted(x for x in os.listdir(self.wf_dir) if x.endswith(".npy"))
        self.sta_lat_min = float(sta_lat_min)
        self.sta_lat_max = float(sta_lat_max)
        self.sta_lon_min = float(sta_lon_min)
        self.sta_lon_max = float(sta_lon_max)
        self.src_lat_min = float(src_lat_min)
        self.src_lat_max = float(src_lat_max)
        self.src_lon_min = float(src_lon_min)
        self.src_lon_max = float(src_lon_max)
        self.dep_min = float(dep_min)
        self.dep_max = float(dep_max)
        self.mag_min = float(mag_min)
        self.mag_max = float(mag_max)
        self.min_get = float(min_get)
        self.max_get = float(max_get)
        self.target_length = int(target_length)
        self.waveform_mean_value = float(waveform_mean_value)
        self.waveform_std_value = float(waveform_std_value)
        self.abs_geo_encoder_type = str(abs_geo_encoder_type).lower()
        self.epi_dist_min = float(epi_dist_min)
        self.epi_dist_max = float(epi_dist_max)

    def convert_to_range(self, value, min_num, max_num, label_name):
        return project_condition_value(value, min_num, max_num, self.min_get, self.max_get, label_name)

    def _load_waveform(self, wf_item_path):
        wf = np.load(wf_item_path).astype(np.float32, copy=False)
        expected_shape = (self.target_length, EXPECTED_WAVEFORM_CHANNELS)
        if wf.shape != expected_shape:
            raise ValueError(
                f"Waveform shape mismatch for {wf_item_path}: got {wf.shape}, expected {expected_shape}"
            )
        return normalize_waveform_array(
            wf,
            mean_value=self.waveform_mean_value,
            std_value=self.waveform_std_value,
        )

    def __getitem__(self, idx):
        wf_name = self.wf_path[idx]
        wf_item_path = os.path.join(self.wf_dir, wf_name)
        wf = self._load_waveform(wf_item_path)
        wf_tensor = torch.from_numpy(wf).float().permute(1, 0)

        parts = wf_name[:-4].split("_")
        if len(parts) < 6:
            raise ValueError(f"Waveform filename does not contain enough label fields: {wf_name}")
        sta_lat, sta_lon, src_lat, src_lon, dep, mag = parts[:6]
        labels = [
            self.convert_to_range(sta_lat, self.sta_lat_min, self.sta_lat_max, "sta_lat"),
            self.convert_to_range(sta_lon, self.sta_lon_min, self.sta_lon_max, "sta_lon"),
            self.convert_to_range(src_lat, self.src_lat_min, self.src_lat_max, "src_lat"),
            self.convert_to_range(src_lon, self.src_lon_min, self.src_lon_max, "src_lon"),
            self.convert_to_range(dep, self.dep_min, self.dep_max, "dep"),
            self.convert_to_range(mag, self.mag_min, self.mag_max, "mag"),
        ]
        if uses_epicentral_distance(self.abs_geo_encoder_type):
            distance_km = haversine_km(float(sta_lat), float(sta_lon), float(src_lat), float(src_lon))
            labels.append(self.convert_to_range(distance_km, self.epi_dist_min, self.epi_dist_max, "epi_dist"))
        elif uses_relative_geometry(self.abs_geo_encoder_type):
            labels.extend(
                encode_relative_geometry(
                    sta_lat=float(sta_lat),
                    sta_lon=float(sta_lon),
                    src_lat=float(src_lat),
                    src_lon=float(src_lon),
                    distance_max_km=self.epi_dist_max,
                    output_min=self.min_get,
                    output_max=self.max_get,
                )
            )
        label = torch.tensor(labels, dtype=torch.float32)
        return wf_tensor, label

    def __len__(self):
        return len(self.wf_path)


if __name__ == "__main__":
    wf_dir = "../train_stead_data"
    sta_lat_min, sta_lat_max = 18.0, 55.0
    sta_lon_min, sta_lon_max = 122.0, 156.0
    src_lat_min, src_lat_max = 18.0, 55.0
    src_lon_min, src_lon_max = 122.0, 156.0
    dep_min, dep_max = -100.0, 800.0
    mag_min, mag_max = -1.0, 10.0
    min_get, max_get = 0, 1000
    target_length = 3000
    waveform_mean_value = 0.0
    waveform_std_value = 1.0
    dataset = LoadWaveform(
        wf_dir,
        sta_lat_min,
        sta_lat_max,
        sta_lon_min,
        sta_lon_max,
        src_lat_min,
        src_lat_max,
        src_lon_min,
        src_lon_max,
        dep_min,
        dep_max,
        mag_min,
        mag_max,
        min_get,
        max_get,
        target_length,
        waveform_mean_value,
        waveform_std_value,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True, pin_memory=True
    )
