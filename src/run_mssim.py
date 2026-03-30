import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms.functional as TF
from tqdm import tqdm


class ImageOnlyDataset(Dataset):
    def __init__(self, df, size=(256, 256)):
        self.df = df
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["path"]
        image = read_image(file_name, mode=ImageReadMode.RGB).float() / 255.0
        image = TF.resize(image, self.size)
        image = torch.clamp(image, min=0.0, max=1.0)
        return image

# https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py 

def _gaussian_window(window_size, sigma, channel, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    window_2d = window_2d / window_2d.sum()
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, data_range, k1=0.01, k2=0.03):
    c = img1.shape[1]
    padding = window.shape[-1] // 2
    mu1 = F.conv2d(img1, window, padding=padding, groups=c)
    mu2 = F.conv2d(img2, window, padding=padding, groups=c)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=c) - mu1_mu2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def ms_ssim(img1, img2, data_range=1.0, window_size=11, window_sigma=1.5):
    weights = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        device=img1.device,
        dtype=img1.dtype,
    )
    levels = weights.shape[0]
    mcs = []
    window = _gaussian_window(
        window_size, window_sigma, img1.shape[1], img1.device, img1.dtype
    )
    for i in range(levels):
        ssim_val, cs_val = _ssim(img1, img2, window, data_range)
        if i < levels - 1:
            mcs.append(cs_val)
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
    mcs = torch.stack(mcs, dim=0)
    return torch.prod(mcs ** weights[:-1].view(-1, 1), dim=0) * (
        ssim_val ** weights[-1]
    )


def compute_ms_ssim(
    real_images_df, synth_images_df, batch_size=32, num_workers=0, device="cuda"
):
    real_dataset = ImageOnlyDataset(df=real_images_df)
    synth_dataset = ImageOnlyDataset(df=synth_images_df)

    real_dataloader = DataLoader(
        real_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    synth_dataloader = DataLoader(
        synth_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    scores = []
    with torch.no_grad():
        for real_batch, synth_batch in zip(real_dataloader, synth_dataloader):
            real_batch = real_batch.to(device)
            synth_batch = synth_batch.to(device)
            batch_scores = ms_ssim(real_batch, synth_batch, data_range=1.0)
            scores.extend(batch_scores.detach().cpu().tolist())
    return float(np.mean(scores))


@dataclass(frozen=True)
class PairSpec:
    name: str
    label: str
    a_name: str
    b_name: str
    a_df: pd.DataFrame
    b_df: pd.DataFrame
    control: bool = False


def sample_two_disjoint(df, n_samples, rng):
    if len(df) < 2 * n_samples:
        raise ValueError(f"Need >= {2 * n_samples} rows, got {len(df)}")
    idx = rng.choice(len(df), size=2 * n_samples, replace=False)
    a_idx = idx[:n_samples]
    b_idx = idx[n_samples:]
    a = df.iloc[a_idx].reset_index(drop=True)
    b = df.iloc[b_idx].reset_index(drop=True)
    return a, b


def sample_one(df, n_samples, rng):
    if len(df) < n_samples:
        raise ValueError(f"Need >= {n_samples} rows, got {len(df)}")
    idx = rng.choice(len(df), size=n_samples, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def load_existing_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compute MS-SSIM and BioVIL CLIP-score pairs."
    )
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="msssim_results.json",
        help="JSON output path (relative to repo root).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by loading existing JSON and continuing repeats.",
    )
    parser.add_argument(
        "--skip-ms-ssim",
        action="store_true",
        help="Skip MS-SSIM computation.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP-score computation.",
    )
    parser.add_argument(
        "--biovil-ckpt",
        type=str,
        default=None,
        help="Path to BioViL image encoder checkpoint.",
    )
    parser.add_argument(
        "--biovil-model-id",
        type=str,
        default="microsoft/BiomedVLP-BioViL-T",
        help="HuggingFace model id for BioViL text model.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    measurement_df_real = pd.read_csv("anatomical_plausibility_signals.csv")
    measurement_df_cheff = pd.read_csv("anatomical_plausibility_signals_cheff.csv")
    measurement_df_roentgen = pd.read_csv(
        "anatomical_plausibility_signals_roentgen.csv"
    )
    measurement_df_chexpert = pd.read_csv("morphometric_measurements_chexpert.csv")

    nofinding_cases_real = measurement_df_real[
        measurement_df_real["prompt"].str.contains("healthy", na=False, case=False)
    ]
    cardiomegaly_cases_real = measurement_df_real[
        measurement_df_real["prompt"].str.contains(
            "with cardiomegaly", na=False, case=False
        )
    ]

    nofinding_cases_cheff = measurement_df_cheff[
        measurement_df_cheff["prompt"].str.contains("healthy", na=False, case=False)
    ]
    cardiomegaly_cases_cheff = measurement_df_cheff[
        measurement_df_cheff["prompt"].str.contains(
            "with cardiomegaly", na=False, case=False
        )
    ]

    nofinding_cases_roentgen = measurement_df_roentgen[
        measurement_df_roentgen["prompt"].str.contains("healthy", na=False, case=False)
    ]
    cardiomegaly_cases_roentgen = measurement_df_roentgen[
        measurement_df_roentgen["prompt"].str.contains(
            "with cardiomegaly", na=False, case=False
        )
    ]
    no_finding_cases_chexpert = measurement_df_chexpert[
        measurement_df_chexpert["No Finding"] == 1.0
    ]

    cardiomegaly_cases_chexpert = measurement_df_chexpert[
        measurement_df_chexpert["Cardiomegaly"] == 1.0
    ]

    pairs = [
        PairSpec(
            name="real_vs_chexpert",
            label="cardiomegaly",
            a_name="real",
            b_name="chexpert",
            a_df=cardiomegaly_cases_real,
            b_df=cardiomegaly_cases_chexpert,
            control=False,
        ),
        PairSpec(
            name="real_vs_chexpert",
            label="no_finding",
            a_name="real",
            b_name="chexpert",
            a_df=nofinding_cases_real,
            b_df=no_finding_cases_chexpert,
            control=False,
        ),
        PairSpec(
            name="real_vs_cheff",
            label="no_finding",
            a_name="real",
            b_name="cheff",
            a_df=nofinding_cases_real,
            b_df=nofinding_cases_cheff,
            control=False,
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="no_finding",
            a_name="real",
            b_name="roentgen",
            a_df=nofinding_cases_real,
            b_df=nofinding_cases_roentgen,
            control=False,
        ),
        PairSpec(
            name="real_vs_cheff",
            label="cardiomegaly",
            a_name="real",
            b_name="cheff",
            a_df=cardiomegaly_cases_real,
            b_df=cardiomegaly_cases_cheff,
            control=False,
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="cardiomegaly",
            a_name="real",
            b_name="roentgen",
            a_df=cardiomegaly_cases_real,
            b_df=cardiomegaly_cases_roentgen,
            control=False,
        ),
        PairSpec(
            name="real_vs_real",
            label="no_finding",
            a_name="real",
            b_name="real",
            a_df=nofinding_cases_real,
            b_df=nofinding_cases_real,
            control=True,
        ),
        PairSpec(
            name="real_vs_real",
            label="cardiomegaly",
            a_name="real",
            b_name="real",
            a_df=cardiomegaly_cases_real,
            b_df=cardiomegaly_cases_real,
            control=True,
        ),
        PairSpec(
            name="real_vs_real",
            label="no_finding_cardiomegaly",
            a_name="real",
            b_name="real",
            a_df=nofinding_cases_real,
            b_df=cardiomegaly_cases_real,
            control=False,
        ),
        PairSpec(
            name="real_vs_cheff",
            label="no_finding_cardiomegaly",
            a_name="real",
            b_name="cheff",
            a_df=nofinding_cases_real,
            b_df=cardiomegaly_cases_cheff,
            control=False,
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="no_finding_cardiomegaly",
            a_name="real",
            b_name="roentgen",
            a_df=nofinding_cases_real,
            b_df=cardiomegaly_cases_roentgen,
            control=False,
        ),
    ]

    rng = np.random.default_rng(args.seed)

    results = {}
    if args.resume:
        results = load_existing_json(args.output)

    for pair in pairs:
        desc = f"{pair.name}:{pair.label}"
        key = f"{pair.name}__{pair.label}"
        if key not in results:
            results[key] = {
                "pair_name": pair.name,
                "label": pair.label,
                "a_name": pair.a_name,
                "b_name": pair.b_name,
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": device,
                "ms_ssim_scores": [],
            }
        existing_scores = results[key]["ms_ssim_scores"]
        start_idx = len(existing_scores)
        for repeat_idx in tqdm(range(args.n_repeats), desc=desc):
            if repeat_idx < start_idx:
                continue

            if pair.control:
                a_df, b_df = sample_two_disjoint(pair.a_df, args.n_samples, rng)
            else:
                a_df = sample_one(pair.a_df, args.n_samples, rng)
                b_df = sample_one(pair.b_df, args.n_samples, rng)

            if args.skip_ms_ssim:
                ms_ssim_score = None
            else:
                ms_ssim_score = compute_ms_ssim(
                    a_df,
                    b_df,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                )

            results[key]["ms_ssim_scores"].append(ms_ssim_score)

        # persist after each pair to avoid losing progress
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
