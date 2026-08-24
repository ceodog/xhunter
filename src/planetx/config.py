"""Prior configuration: loading and sampling from configs/prior.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Distribution:
    dist: str
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        if self.dist == "uniform":
            return float(rng.uniform(self.low, self.high))
        if self.dist == "loguniform":
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        raise ValueError(f"Unknown distribution kind: {self.dist!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "Distribution":
        return cls(dist=d["dist"], low=float(d["low"]), high=float(d["high"]))


@dataclass(frozen=True)
class SimulationConfig:
    n_test_particles: int
    integration_years: float
    dt_years: float
    use_gr: bool = False
    # "rebound" (validated, default) | "secular" | "hybrid" (opt-in,
    # exploratory -- see planetx.simgen.secular's module docstring for the
    # measured accuracy caveats before changing this in a shipped config).
    disk_backend: str = "rebound"


@dataclass(frozen=True)
class PriorConfig:
    theta_hpx: dict[str, Distribution] = field(default_factory=dict)
    nuisance: dict[str, Distribution] = field(default_factory=dict)
    simulation: SimulationConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PriorConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        theta_hpx = {k: Distribution.from_dict(v) for k, v in raw["theta_hpx"].items()}
        nuisance = {k: Distribution.from_dict(v) for k, v in raw["nuisance"].items()}
        sim = SimulationConfig(
            n_test_particles=int(raw["simulation"]["n_test_particles"]),
            integration_years=float(raw["simulation"]["integration_years"]),
            dt_years=float(raw["simulation"]["dt_years"]),
            use_gr=bool(raw["simulation"].get("use_gr", False)),
            disk_backend=str(raw["simulation"].get("disk_backend", "rebound")),
        )
        return cls(theta_hpx=theta_hpx, nuisance=nuisance, simulation=sim)
