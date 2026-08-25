"""Training loop for PosteriorNet on a sharded Parquet dataset produced by
planetx.simgen.orchestrate.generate_dataset.

Known scaffold simplification: theta and object features are trained on raw
physical units (AU, degrees, Earth masses) rather than standardized/whitened
values. For real training runs, standardize each column (and consider log a,
log mass) before this reaches the flow -- raw units make the loss landscape
unnecessarily hard to optimize.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from planetx.constants import OBJECT_FEATURE_KEYS, THETA_KEYS
from planetx.model.posterior_net import PosteriorNet

logger = logging.getLogger(__name__)


class ShardedSimDataset(Dataset):
    """Loads all shards eagerly; fine up to roughly hundreds of thousands of
    simulations. For larger corpora, replace with a streaming IterableDataset
    over the same Parquet shards -- PosteriorNet and collate() don't need to
    change.
    """

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        shards = sorted(data_dir.glob("shard_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no shard_*.parquet files under {data_dir}")
        self.df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        # np.asarray(row["features"], dtype=...) raises "setting an array element
        # with a sequence" when exactly one object was detected: numpy's ragged-
        # array shape inference is ambiguous for a single-element array of arrays
        # (it can't tell a 1-object detection from "flatten this"). Routing through
        # .tolist() first removes the ambiguity -- confirmed against n_objects=0,
        # 1, and >=2 rows from a real GPU-generated shard (orchestrate_gpu.py) that
        # tripped this exact case; the CPU path (orchestrate.py) writes the same
        # schema and was equally exposed, just rarely hit at production's larger
        # disk sizes where "exactly 1 detection" is a rarer outcome.
        features = np.array(row["features"].tolist(), dtype=np.float32)
        if features.size == 0:
            features = features.reshape(0, len(OBJECT_FEATURE_KEYS))
        else:
            features = features.reshape(-1, len(OBJECT_FEATURE_KEYS))
        return {
            "theta": np.asarray(row["theta"], dtype=np.float32),
            "features": features,
            "survey_meta": np.asarray(row["survey_meta"], dtype=np.float32),
        }


def collate(batch: list[dict]) -> dict:
    max_n = max((item["features"].shape[0] for item in batch), default=0)
    max_n = max(max_n, 1)  # guard against an all-empty batch
    feat_dim = len(OBJECT_FEATURE_KEYS)

    feats = np.zeros((len(batch), max_n, feat_dim), dtype=np.float32)
    mask = np.ones((len(batch), max_n), dtype=bool)
    theta = np.zeros((len(batch), len(THETA_KEYS)), dtype=np.float32)
    survey_meta = np.zeros((len(batch), batch[0]["survey_meta"].shape[0]), dtype=np.float32)

    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        if n:
            feats[i, :n] = item["features"]
            mask[i, :n] = False
        theta[i] = item["theta"]
        survey_meta[i] = item["survey_meta"]

    return {
        "features": torch.from_numpy(feats),
        "key_padding_mask": torch.from_numpy(mask),
        "theta": torch.from_numpy(theta),
        "survey_meta": torch.from_numpy(survey_meta),
    }


def train(
    data_dir: str | Path,
    out_path: str | Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> PosteriorNet:
    dataset = ShardedSimDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    net = PosteriorNet(
        object_feature_dim=len(OBJECT_FEATURE_KEYS), theta_dim=len(THETA_KEYS)
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        total, n_batches = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            loss = net.loss(
                batch["features"].to(device),
                batch["key_padding_mask"].to(device),
                batch["survey_meta"].to(device),
                batch["theta"].to(device),
            )
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        logger.info("epoch %d/%d  mean NLL = %.4f", epoch + 1, epochs, total / max(n_batches, 1))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    return net
