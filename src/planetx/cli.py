"""Command-line entry points wiring the three subsystems together:

  planetx simgen run       -- generate a sharded Parquet training set
  planetx obsdata build-x  -- pull real elements into network-ready x
  planetx model train      -- train PosteriorNet (NPE) on a training set
  planetx model infer      -- posterior samples for a real x
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np

from planetx.config import PriorConfig
from planetx.constants import OBJECT_FEATURE_KEYS, THETA_KEYS


@click.group()
def main() -> None:
    """planetx: amortized simulation-based inference for a hypothetical Planet X."""
    # Library modules (simgen.orchestrate, model.train) log via
    # logging.getLogger(__name__) rather than print(), so this is the one
    # place output actually needs to be turned on -- format matches the
    # bare progress-message style those print() calls used to have.
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------- simgen ---
@main.group()
def simgen() -> None:
    """Training-data generation (REBOUND + survey selection forward model)."""


@simgen.command("run")
@click.option("--prior", "prior_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", required=True, type=click.Path())
@click.option("--n-sims", default=1000, show_default=True)
@click.option("--shard-size", default=500, show_default=True)
@click.option("--workers", default=4, show_default=True)
@click.option(
    "--seed0", default=-1, show_default=True,
    help="base seed (sim k uses seed0+k). Negative (default) derives it from the "
         "machine's current timestamp instead, so repeated runs generate different "
         "datasets -- the resolved value is logged so a specific run can be reproduced "
         "later by passing it back explicitly. Pass a non-negative integer for a fully "
         "reproducible run.",
)
@click.option(
    "--progress-bar", "use_tqdm", is_flag=True, default=False,
    help="use a tqdm progress bar (interactive terminals) instead of periodic "
         "1%-interval log lines (better for redirected/cloud-captured logs, the default)",
)
def simgen_run(prior_path, out_dir, n_sims, shard_size, workers, seed0, use_tqdm) -> None:
    from planetx.simgen.orchestrate import generate_dataset

    prior_config = PriorConfig.from_yaml(prior_path)
    generate_dataset(
        prior_config=prior_config, out_dir=out_dir, n_simulations=n_sims,
        shard_size=shard_size, n_workers=workers, seed0=seed0, use_tqdm=use_tqdm,
    )


# --------------------------------------------------------------- obsdata ---
@main.group()
def obsdata() -> None:
    """Real observational data extraction."""


@obsdata.command("build-x")
@click.option("--designations", required=True, help="comma-separated object designations")
@click.option("--out", "out_path", required=True, type=click.Path())
@click.option(
    "--sky-fraction", default=1.0, show_default=True,
    help="placeholder survey metadata -- replace with a real survey's characterization",
)
@click.option("--limiting-mag", default=24.5, show_default=True)
@click.option("--tracking-efficiency", default=1.0, show_default=True)
def obsdata_build_x(designations, out_path, sky_fraction, limiting_mag, tracking_efficiency) -> None:
    from planetx.obsdata.build_x import build_x_from_designations

    click.echo(
        "warning: --sky-fraction/--limiting-mag/--tracking-efficiency are placeholders; "
        "for a real inference run, supply selection_function_metadata() from a "
        "characterized survey connector (see planetx.obsdata.fetch)",
        err=True,
    )
    designation_list = [d.strip() for d in designations.split(",") if d.strip()]
    survey_meta = {
        "sky_fraction": sky_fraction,
        "limiting_mag": limiting_mag,
        "tracking_efficiency": tracking_efficiency,
    }
    fset = build_x_from_designations(designation_list, survey_meta)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, features=fset.features, survey_meta=fset.survey_meta, n_objects=fset.n_objects)
    click.echo(f"wrote x for {fset.n_objects} objects -> {out_path}")


# ----------------------------------------------------------------- model ---
@main.group()
def model() -> None:
    """Train and run the transformer-based posterior network."""


@model.command("train")
@click.option("--data-dir", required=True, type=click.Path(exists=True))
@click.option("--out", "out_path", required=True, type=click.Path())
@click.option("--epochs", default=50, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--device", default="cpu", show_default=True)
def model_train(data_dir, out_path, epochs, batch_size, lr, device) -> None:
    from planetx.model.train import train

    train(
        data_dir=data_dir, out_path=out_path, epochs=epochs,
        batch_size=batch_size, lr=lr, device=device,
    )


@model.command("infer")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--x", "x_path", required=True, type=click.Path(exists=True), help=".npz from obsdata build-x")
@click.option("--n-samples", default=2000, show_default=True)
@click.option("--out", "out_path", required=True, type=click.Path())
def model_infer(checkpoint, x_path, n_samples, out_path) -> None:
    import torch

    from planetx.model.posterior_net import PosteriorNet

    data = np.load(x_path)
    features = torch.from_numpy(data["features"].astype(np.float32)).unsqueeze(0)
    survey_meta = torch.from_numpy(data["survey_meta"].astype(np.float32)).unsqueeze(0)
    mask = torch.zeros(1, features.shape[1], dtype=torch.bool)

    net = PosteriorNet(object_feature_dim=len(OBJECT_FEATURE_KEYS), theta_dim=len(THETA_KEYS))
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.eval()

    samples = net.posterior_samples(features, mask, survey_meta, n=n_samples).squeeze(1).numpy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, samples)

    summary = {k: {"mean": float(samples[:, i].mean()), "std": float(samples[:, i].std())}
               for i, k in enumerate(THETA_KEYS)}
    click.echo(json.dumps(summary, indent=2))
    click.echo(f"wrote {n_samples} posterior samples -> {out_path}")


if __name__ == "__main__":
    main()
