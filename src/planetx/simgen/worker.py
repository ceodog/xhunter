"""One REBOUND simulation: known giant planets (fixed) + HPX (theta) +
a primordial TNO test-particle disk (shaped by nuisance params), integrated
to a single final epoch.

Per the project's design decision (see README.md, "Single-epoch snapshots"),
each simulation yields exactly one (x, theta) training pair: the state at the
end of the integration, not a trajectory. Time evolution happens only inside
this function; nothing downstream ever sees it.
"""

from __future__ import annotations

import numpy as np
import rebound

from planetx.constants import EARTH_MASS_IN_MSUN, GIANT_PLANETS, THETA_KEYS
from planetx.simgen import secular


def _add_giant_planets(sim: "rebound.Simulation", rng: np.random.Generator) -> None:
    """a, e, inc, Omega, omega stay fixed at GIANT_PLANETS' verified J2000
    values -- that's the well-constrained, dynamically important
    architecture (see README.md, "Fixed constants vs. nuisance parameters
    vs. theta"). M (orbital phase) is drawn uniformly per giant per
    simulation instead of using GIANT_PLANETS' J2000 M: t=0 doesn't
    correspond to a real calendar epoch anyway (see "The primordial disk
    and dynamical relaxation"), so pinning every simulation to today's
    specific orbital phase for each giant would be an arbitrary choice with
    no cost to removing it, rather than a physically meaningful one.
    """
    for name, el in GIANT_PLANETS.items():
        sim.add(
            m=el["m"], a=el["a"], e=el["e"], inc=np.radians(el["inc"]),
            Omega=np.radians(el["Omega"]), omega=np.radians(el["omega"]),
            M=rng.uniform(0, 2 * np.pi), name=name,
        )


def _add_hpx(sim: "rebound.Simulation", theta: dict[str, float]) -> None:
    sim.add(
        m=theta["mass"] * EARTH_MASS_IN_MSUN,
        a=theta["a"], e=theta["e"], inc=np.radians(theta["i"]),
        Omega=np.radians(theta["Omega"]), omega=np.radians(theta["omega"]),
        M=np.radians(theta["M"]), name="hpx",
    )


def _add_primordial_disk(
    sim: "rebound.Simulation", nuisance: dict[str, float], n: int, rng: np.random.Generator
) -> list[str]:
    """Massless test particles standing in for the primordial Kuiper belt.

    Being massless, they neither perturb each other nor the massive bodies,
    so this loop is embarrassingly cheap relative to the massive-body
    integration itself.
    """
    hashes = []
    for idx in range(n):
        a = rng.uniform(nuisance["disk_inner_edge"], nuisance["disk_outer_edge"])
        e = float(np.clip(rng.rayleigh(nuisance["disk_e_scale"]), 0.0, 0.9))
        inc = float(np.clip(rng.rayleigh(nuisance["disk_i_scale"]), 0.0, 60.0))
        h = f"tno_{idx}"
        sim.add(
            m=0.0, a=a, e=e, inc=np.radians(inc),
            Omega=rng.uniform(0, 2 * np.pi), omega=rng.uniform(0, 2 * np.pi),
            M=rng.uniform(0, 2 * np.pi), name=h,
        )
        hashes.append(h)
    return hashes


def _sample_primordial_disk(nuisance: dict[str, float], n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Same distributions as _add_primordial_disk, but returns arrays
    instead of adding particles to a REBOUND sim -- used by disk_backend
    "secular"/"hybrid", which need to inspect/route particles before (if at
    all) adding them to REBOUND. Uses vectorized rng calls, so it does NOT
    reproduce _add_primordial_disk's per-particle values for the same seed
    (different draw order) -- not a concern, since these backends are a
    different computational method, not required to bit-match "rebound"."""
    a = rng.uniform(nuisance["disk_inner_edge"], nuisance["disk_outer_edge"], n)
    e = np.clip(rng.rayleigh(nuisance["disk_e_scale"], n), 0.0, 0.9)
    inc = np.radians(np.clip(rng.rayleigh(nuisance["disk_i_scale"], n), 0.0, 60.0))
    Omega = rng.uniform(0, 2 * np.pi, n)
    omega = rng.uniform(0, 2 * np.pi, n)
    M = rng.uniform(0, 2 * np.pi, n)
    return {"a": a, "e": e, "inc": inc, "Omega": Omega, "omega": omega, "M": M}


def _maybe_add_gr(sim: "rebound.Simulation") -> None:
    """Optional 1-PN correction. gr_full is velocity-dependent and requires
    IAS15 (WHFast does not support it) -- see REBOUNDx docs. Left disabled by
    default; enable via configs/prior.yaml `simulation.use_gr: true`.
    """
    import reboundx  # local import: keep reboundx optional at module load time

    sim.integrator = "ias15"
    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr_full")
    rebx.add_force(gr)
    gr.params["c"] = 63241.08  # AU/yr, speed of light in sim units
    return rebx  # keep a reference alive for the caller


def run_one(
    theta: dict[str, float],
    nuisance: dict[str, float],
    n_test_particles: int,
    integration_years: float,
    dt_years: float,
    use_gr: bool,
    seed: int,
    disk_backend: str = "rebound",
) -> dict:
    """Run a single hypothetical-Solar-System simulation and return its final state.

    disk_backend selects how the massless test-particle disk is propagated
    (the Sun + 4 giants + HPX always integrate via real REBOUND N-body,
    regardless of this setting -- see planetx.simgen.secular's module
    docstring for why):
      "rebound" (default): every test particle integrated by REBOUND.
        Validated, expensive -- the only backend used in any shipped config.
      "secular": every test particle propagated by closed-form linear
        secular theory (planetx.simgen.secular). ~8800x cheaper, but
        measurably worse accuracy, especially for particles inside a
        Neptune mean-motion resonance -- see secular.py's module docstring
        for the caveats. Opt-in / exploratory only.
      "hybrid": resonant test particles routed to real N-body, the rest use
        the closed form. ~14-16x cheaper than "rebound" at this project's
        production scale; same caveats as "secular", less severe since the
        resonant subset is excluded. Opt-in / exploratory only.

    Returns a dict with:
      theta:        the sampled HPX parameter dict (the training label)
      hpx_final:    HPX's own elements at the final epoch
      tnos:         list of dicts, one per surviving test particle, with
                    keys a, e, i (deg), Omega (deg), omega (deg)
    """
    if disk_backend not in ("rebound", "secular", "hybrid"):
        raise ValueError(f"Unknown disk_backend: {disk_backend!r} (expected 'rebound', 'secular', or 'hybrid')")

    rng = np.random.default_rng(seed)

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0, name="sun")
    _add_giant_planets(sim, rng)
    _add_hpx(sim, theta)
    # Every particle added above this line is massive; N_active tells
    # REBOUND's gravity routine to only compute forces FROM these bodies,
    # not between the massless test particles added next. Without this,
    # REBOUND's default "basic" gravity module loops over all N^2 particle
    # pairs regardless of mass, making the primordial disk's cost scale
    # quadratically in n_test_particles instead of linearly -- empirically,
    # ~33x slower at n_test_particles=2000 on this project's benchmark.
    sim.N_active = len(sim.particles)

    disk = None
    disk_resonant = None
    if disk_backend == "rebound":
        _add_primordial_disk(sim, nuisance, n_test_particles, rng)
    else:
        disk = _sample_primordial_disk(nuisance, n_test_particles, rng)
        if disk_backend == "hybrid":
            disk_resonant = secular.resonant_mask(disk["a"], disk["e"])
            for idx in np.flatnonzero(disk_resonant):
                sim.add(
                    m=0.0, a=disk["a"][idx], e=disk["e"][idx], inc=disk["inc"][idx],
                    Omega=disk["Omega"][idx], omega=disk["omega"][idx], M=disk["M"][idx],
                    name=f"tno_{idx}",
                )

    sim.move_to_com()

    rebx_ref = None
    if use_gr:
        rebx_ref = _maybe_add_gr(sim)  # noqa: F841 (kept alive for integrator lifetime)
    else:
        sim.integrator = "whfast"
        # safe_mode defaults to 1, which resynchronizes (recomputes Jacobi
        # coordinates from scratch) every single step -- REBOUND's own docs
        # call this "substantially slower... than it can be." safe_mode=0 is
        # safe here specifically because run_one() never reads or modifies
        # particle state mid-integration -- state is read only once, after
        # sim.integrate() returns below, which is exactly the access pattern
        # safe_mode=0 is designed for.
        sim.integrator.safe_mode = 0
        sim.dt = dt_years

    sim.integrate(integration_years)

    hpx = sim.particles["hpx"]
    o = hpx.orbit(primary=sim.particles[0])
    hpx_final = {
        "mass": theta["mass"],
        "a": o.a, "e": o.e, "i": np.degrees(o.inc),
        "Omega": np.degrees(o.Omega) % 360.0, "omega": np.degrees(o.omega) % 360.0,
        "M": np.degrees(o.M) % 360.0,
    }

    tnos = []
    if disk_backend == "rebound":
        for p in sim.particles[5:]:  # sun + 4 giants + hpx = indices 0..5
            if p.name == "hpx":
                continue
            try:
                o = p.orbit(primary=sim.particles[0])
            except RuntimeError:
                continue  # unbound / ejected particle
            if o.a <= 0 or o.e >= 1.0:
                continue  # ejected or hyperbolic
            tnos.append({
                "a": o.a, "e": o.e, "i": np.degrees(o.inc),
                "Omega": np.degrees(o.Omega) % 360.0, "omega": np.degrees(o.omega) % 360.0,
                "M": np.degrees(o.M) % 360.0,
            })
    else:
        if disk_backend == "hybrid":
            resonant_indices = np.flatnonzero(disk_resonant)
            for k, idx in enumerate(resonant_indices):
                p = sim.particles[6 + k]  # sun + 4 giants + hpx = indices 0..5
                try:
                    o = p.orbit(primary=sim.particles[0])
                except RuntimeError:
                    continue
                if o.a <= 0 or o.e >= 1.0:
                    continue
                tnos.append({
                    "a": o.a, "e": o.e, "i": np.degrees(o.inc),
                    "Omega": np.degrees(o.Omega) % 360.0, "omega": np.degrees(o.omega) % 360.0,
                    "M": np.degrees(o.M) % 360.0,
                })
            secular_indices = np.flatnonzero(~disk_resonant)
        else:  # "secular"
            secular_indices = np.arange(n_test_particles)

        eigsys = secular.EigenSystem(secular.massive_body_elements(theta))
        tnos.extend(secular.propagate_disk(eigsys, disk, secular_indices, integration_years))

    return {"theta": {k: theta[k] for k in THETA_KEYS}, "hpx_final": hpx_final, "tnos": tnos}
