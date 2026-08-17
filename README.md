# planetx-hunter

Amortized simulation-based inference for a hypothetical Planet X (HPX),
using REBOUND-simulated Solar Systems and a transformer-based neural
posterior estimator over real TNO/ETNO populations.

## The idea, precisely stated

This is **not** a supervised regression project ("predict Planet X's orbit
from TNO data"). It's **amortized Bayesian simulation-based inference
(SBI)**, specifically **Neural Posterior Estimation (NPE)**: we can't write
down a tractable likelihood for "given HPX's parameters, what TNO population
would we see," but we *can* simulate it with REBOUND. NPE trains a
conditional density estimator on (simulation, label) pairs so that, at
convergence, it approximates the true Bayesian posterior for *any* input —
not just a point estimate, and not just the simulations it was trained on.

### θ and x

- **θ** — the inference target: HPX's own orbital elements + mass
  `(mass, a, e, i, Ω, ω, M)`. `M` (orbital phase) is expected to come out
  with a nearly flat posterior — that's physically correct, not a bug.
- **x** — the evidence: the set of TNO/ETNO orbital elements (+
  uncertainty) that a real survey would actually report, after passing
  through that survey's detection biases. **x is a set, not a sequence** —
  object order carries no meaning and population size varies, so every
  network component that consumes x must be permutation-invariant.

Both θ and x are made of the same vocabulary (orbital elements), just for
different bodies: θ describes the one unseen perturber, x describes the
many already-known bodies it perturbs.

### Single-epoch snapshots, not time series

Inside one REBOUND run, every body's elements evolve continuously over the
~Gyr integration — that evolution is essential physics (it's what builds up
the resonant/secular clustering signature), but it is never exposed to the
network as an input axis. Real survey data is a snapshot at one epoch
("now"); decades of astrometric baseline (even extended via precovery)
resolve *that one* dynamical state more precisely, not a second, different
state — the precession/libration timescales that matter here are Myr–Gyr,
far beyond human observing baselines. So:

- Each simulation is integrated to a single final epoch (`age of the
  system`), and **one (x, θ) pair** is extracted at that epoch.
- Training on snapshots from multiple *simulations* is fine and useful
  (different runs reach different dynamical stages); training on multiple
  *time points from the same real system* is not possible with real data
  and is not what this pipeline does.

(What "final epoch" actually corresponds to in calendar terms is a separate
question with a real subtlety — see "The primordial disk and dynamical
relaxation" under Physical Principles below.)

### Feature parity

The exact same feature-engineering code
(`planetx.featurelib.build_feature_set`) is called from both the
simulation pipeline (`planetx.simgen.selection`) and the real-data
pipeline (`planetx.obsdata.build_x`). This is what prevents train/inference
skew — if the two sides ever computed features differently, the network's
posterior on real data would be unreliable regardless of how well training
converged.

## Physical principles and assumptions

### Osculating orbital elements: the shared representation

Both θ and every row of x are **osculating elements** `(a, e, i, Ω, ω, M)` —
the parameters of the two-body Keplerian ellipse that matches a body's true
position and velocity at one instant, even though the real trajectory is
perturbed (non-Keplerian) by other bodies' gravity. ("Osculating," from
Latin for "kissing": the ellipse touches and matches the true orbit at
exactly that instant, then diverges.) This is why simulated and real data
can share one feature schema: `hpx.orbit(primary=...)` in `worker.py`
computes osculating elements from a Cartesian N-body state the same way a
real orbit determination computes them from astrometric observations — the
same six-parameter description, opposite direction of computation.

### Fixed constants vs. nuisance parameters vs. θ

Three different roles, easy to conflate:

| | Sampled per simulation? | Marginalized? | Example |
|---|---|---|---|
| θ | yes | no — this is the label | HPX mass, a, e, i, Ω, ω, M |
| nuisance | yes | yes (never a label) | primordial disk mass/extent, disk excitation |
| fixed | no — same every run | n/a | giant planet masses, a, e, inc, Ω, ω |

The giant planets' masses and dynamical architecture (`a, e, inc, Ω, ω`)
are known to far higher precision (spacecraft tracking, DE440/DE441) than
would meaningfully affect the HPX signal, so they're held fixed
(`planetx.constants.GIANT_PLANETS`) rather than sampled as a nuisance
latent — sampling them would waste model capacity for no statistical
benefit. Primordial disk properties, by contrast, are genuinely uncertain
and could confound HPX's effect on the TNO population if not marginalized,
so they *are* sampled every run and never fed back as a label.

One element doesn't fit cleanly into either row: each giant's orbital
**phase** (`M`) is randomized uniformly per simulation directly in
`simgen/worker.py`, rather than being fixed at its J2000 value like the
rest of `GIANT_PLANETS`, or sampled from a configured nuisance prior in
`prior.yaml`. See "The primordial disk and dynamical relaxation" below for
why — in short, `t=0` isn't a real calendar epoch, so there's no principled
reason to prefer today's specific orbital phase for each giant over any
other, and randomizing it is essentially free (unlike the dynamical
architecture, which chaos makes genuinely expensive and ill-posed to
resample — also covered there).

**Provenance note:** `GIANT_PLANETS`' orbital elements (mass ratios aside —
those are well-known IAU constants) were initially written from general
recall rather than a live lookup, and turned out to have real errors when
checked against JPL SSD's actual reference table for this purpose
([Keplerian Elements for Approximate Positions of the Major
Planets](https://ssd.jpl.nasa.gov/planets/approx_pos.html), Standish
1992/2006) — Saturn's eccentricity was off by ~5%, Neptune's by ~24%, plus
smaller errors elsewhere. They've since been corrected against that table
(J2000.0 epoch; `constants.py`'s docstring documents the exact convention
conversion used). If you regenerate this file, don't recall these values
either — go back to that source, or better, a live JPL Horizons/DE440
query.

### Why the inner planets aren't modeled at all

`simgen/worker.py` builds each simulation from the Sun, the four giant
planets, HPX, and the primordial disk -- Mercury, Venus, Earth, and Mars
never appear. This isn't an oversight: at TNO/ETNO distances (tens to
hundreds of AU), a distant body can't resolve the inner solar system's
internal structure -- its effect there is, to extremely high precision,
indistinguishable from just adding the inner planets' mass to the Sun's.
And that combined mass (Mercury + Venus + Earth + Mars ≈ 5.9×10⁻⁶ M☉, about
6 parts per million of the Sun) is roughly 160x smaller than Jupiter alone
(954.79×10⁻⁶ M☉ in `constants.GIANT_PLANETS`), and is comparable to or
smaller than the *low end* of HPX's own hypothesized mass range (1 Earth
mass ≈ 3×10⁻⁶ M☉) -- below the smallest signal this project is trying to
detect. (Worth noting precisely: the current code doesn't even fold that
6 ppm into the Sun's mass, i.e. `sim.add(m=1.0, ...)` rather than
`m=1.0 + 5.9e-6` -- a negligible additional simplification, smaller than
the uncertainty already carried by the nuisance-parameter priors.)

There's also an independent, purely computational reason to leave them
out -- see "Numerical integration validity: the timestep floor" below.
Sun + 4 giant planets (+ a hypothetical distant perturber) is also the
standard setup in the actual Planet Nine dynamics literature (Batygin &
Brown, Nesvorný, etc.) for exactly these reasons -- not a shortcut unique
to this scaffold.

### The primordial disk and dynamical relaxation

`simgen/worker.py`'s `_add_primordial_disk` seeds each simulation with
massless test particles standing in for the early, dynamically "cold" disk
of leftover planetesimals from Solar System formation — before
gravitational scattering by the giant planets (and, in this project's
hypothesis, HPX) sculpted it into the Kuiper Belt / scattered disk / ETNO
structure observed today. Integrating this disk forward under N-body
gravity is what allows realistic resonant/secular structure to develop;
without that Gyr-scale relaxation, there'd be nothing but the disk's
arbitrary initial condition to observe.

A nuance worth being precise about: "final epoch" means θ and x are read
out at the same simulated instant as each other — not that the simulated
clock literally reconstructs real calendar time. Every simulation's `t=0`
is seeded with the giant planets' *current* (J2000) elements
(`constants.GIANT_PLANETS`), so integrating forward by `integration_years`
does not end at real "now"; it ends `integration_years` in the future. See
the `SOLAR_SYSTEM_AGE_YEARS` comment in `constants.py` for the full
reasoning — in short, that duration is chosen as "long enough for
realistic structure to develop," not as a literal replay of solar system
history from an accurate primordial giant-planet configuration (which
would require modeling giant-planet migration itself, e.g. the Nice
model — beyond this project's scope). What does still hold regardless:
θ and x share the same final simulated instant, so they remain a valid,
mutually consistent training pair no matter what that instant means in
calendar terms.

**Why not just backward-integrate today's elements to find the true `t=0`?**
REBOUND supports it (Newtonian gravity is time-reversible), but it wouldn't
recover an accurate primordial configuration, for two independent reasons.
First, the solar system is chaotic: Laskar (1989) found a ~5 Myr Lyapunov
time for the full 8-planet system, and Murray & Holman (1999) found
7–20 Myr for the giant-planet subsystem alone, driven by overlapping
three-body mean-motion resonances among the giants themselves.
`integration_years` (4.5 Gyr) is ~300–600 Lyapunov times, so any
backward-integrated trajectory — even with perfect arithmetic — diverges
completely from the real historical one long before reaching anywhere near
a genuinely primordial epoch; you'd get *a* dynamically valid trajectory,
not *the* real one. Second, and independent of chaos: giant-planet
migration (the mechanism thought to have produced the compact
pre-migration configuration in the first place, per the Nice model above)
is driven by angular-momentum exchange with the primordial disk — so
backward-integrating the giants *without* the disk removes the actual
causal driver, and would just relax into bounded oscillation around
roughly today's configuration rather than reconstructing a genuinely
different pre-migration state. This is also why the actual Nice-model
literature (Tsiganis, Gomes, Morbidelli & Levison 2005; Levison et al.
2011) doesn't try to solve for *the* historical trajectory at all — it
runs forward ensembles from many plausible primordial setups and checks
which ones statistically reproduce today's solar system, the same
simulate-many-hypotheses logic this project already applies to HPX itself,
just one level deeper.

Given that, `a, e, inc, Ω, ω` for the giants are held fixed at their
verified J2000 values rather than resampled — chaos makes nearby
alternatives behave unpredictably rather than smoothly (unlike the disk's
nuisance parameters), and there's no principled distribution to sample
from even if that weren't a problem. Each giant's orbital **phase** (`M`)
is treated differently: since `t=0` isn't a real calendar epoch anyway,
there's no reason to prefer today's specific phase over any other, so
`simgen/worker.py`'s `_add_giant_planets` draws it uniformly per
simulation instead of using `GIANT_PLANETS`' fixed J2000 `M` — free
robustness to an arbitrary detail, without touching the well-constrained
dynamical architecture or reopening the chaos problem above.

### How real orbits (and their uncertainty) are actually determined

A single astrometric observation gives only an angular sky position (RA,
Dec) at one epoch — nowhere near enough to fix a 6-parameter orbit.
Determining one is an inverse problem: given many sky positions spread over
time, find the six elements whose predicted positions best match all of
them (classically via Gauss's/Laplace's method for an initial guess, then
nonlinear least-squares differential correction). The `sigma` values
`obsdata.fetch.query_sbdb` pulls out are the *formal* uncertainties from
that fit's covariance matrix — how tightly constrained each element is
*given the observations actually available*, not a fixed physical constant.
This is why observational arc length matters so much for distant TNOs: an
object with an orbital period of thousands of years barely moves along its
orbit even over decades of astrometry (even extended via precovery), so
short-arc fits leave large formal uncertainties — the concrete mechanism
behind why real epochs, however many calendar years they span, still
resolve only one dynamical state (see "Single-epoch snapshots" above).

### Detectability: apparent vs. absolute magnitude

Absolute magnitude `H` is the brightness a body would have at 1 AU from
both Sun and observer at zero phase angle — a proxy for physical size
(given an assumed albedo), independent of current geometry. It sets
whether an object is detectable at all, via `V = H + 5·log₁₀(r·Δ)`
(heliocentric distance × geocentric distance) — used identically on both
sides: `simgen/selection.py`'s `SimpleSelectionFunction.apply` computes it
to decide whether a simulated object clears the survey's limiting
magnitude, and it's the same physical quantity real surveys use to decide
what gets discovered. Because `r` depends on the object's own `(a, e)` and
its *current* orbital phase `M` (via Kepler's equation,
`_heliocentric_distance`), this correctly reproduces a real, well-known
bias: eccentric objects are far easier to detect near perihelion than near
aphelion, which is why real ETNO discoveries cluster near perihelion — the
same effect at the center of the actual Brown/Batygin-vs-OSSOS clustering
debate.

### The assumed size (brightness) distribution

`selection.py`'s `_sample_H` draws each detected candidate's absolute
magnitude from a single power-law luminosity function
(`dN/dH ∝ 10^(slope·H)`) rather than uniformly — real KBO/TNO populations
have far more small/faint objects than large/bright ones. Range and slope
are calibrated against real JPL SBDB data for 8 securely-classified ETNOs
(Sedna through 2013 FT28, observed H range 1.5–7.2); the slope itself
(`0.1`) is deliberately gentler than commonly-cited literature values
(~0.7–0.9), because those are fit over a much narrower magnitude range in
their source studies and blow up when extrapolated across this project's
wider range (verified empirically — see the comment in `_sample_H`). Using
the *detected* sample's own distribution directly would be circular
reasoning, since it's already brightness-selected by construction.

### The selection function: the load-bearing piece

The real historical Planet Nine controversy (Batygin & Brown's claimed
clustering vs. OSSOS's finding that the same clustering is reproduced by
survey bias alone) is exactly the failure mode this pipeline has to guard
against. NPE's validity guarantee — that the trained network converges to
the *true* posterior — only holds if the distribution of simulated x
matches how real x is actually generated, biases included. Concretely:
every simulated population must be passed through a **survey selection
forward model** before it becomes a training x
(`planetx.simgen.selection`), and real inference must only use catalogs
from surveys with a **published, characterized selection function**
(OSSOS, DES, Rubin/Sorcha) — never an uncharacterized compilation like the
raw MPC database.

### Two filters, two different physical mechanisms

Not every initial test particle ends up as a row in x. It passes through
two independent stages:

1. **Dynamical survival** (`worker.py`, real N-body physics) — over a
   multi-Gyr integration, gravitational scattering genuinely ejects some
   fraction of the disk (`o.a <= 0 or o.e >= 1.0` → dropped). This is a
   real physical process, not a modeling artifact.
2. **The selection function** (`selection.py`, observational bias, *not*
   the body's own physics) — of the dynamical survivors, only those
   clearing sky-coverage, magnitude, and tracking-efficiency cuts become a
   row in x. See "The selection function: the load-bearing piece" above
   for why realism here is critical, not optional.

### Gravity computation: why `N_active` matters physically

REBOUND's default gravity routine loops over particle pairs regardless of
mass; without telling it that only the Sun/giants/HPX are massive
(`sim.N_active = len(sim.particles)`, set right after adding them and
before the massless disk), it computes forces for all 2000×2000
test-particle-vs-test-particle pairs too — each contributing exactly zero
physics (massless particles exert no gravity) but still costing compute.
This isn't just a performance detail: it reflects the actual physical
assumption the model makes (a massless test-particle disk that doesn't
self-gravitate), and `N_active` is what tells REBOUND to compute forces
consistent with that assumption instead of wasting cycles re-deriving zero.

### Numerical integration validity: the timestep floor

WHFast (a symplectic integrator) requires `dt` well below the shortest
orbital period in the system — the standard rule of thumb is
`dt <= P_min/20` — or it doesn't just lose precision, it aliases the
fastest orbit into numerically meaningless dynamics. See the `dt_years`
comment in `configs/prior.yaml` for the Jupiter-driven derivation of the
current `0.5` year value. This same rule is part of why Mercury etc. are
excluded (see "Why the inner planets aren't modeled at all" above): their
short periods would force `dt` down by another ~40x, for physics smaller
than the noise floor elsewhere in the model.

### Known physical asymmetries and unvalidated assumptions

- **Synthetic vs. real `sigma` come from different physical processes.** On
  the simulated side, `sigma` is a hand-built heuristic (`_uncertainty_for`,
  scaling with apparent magnitude). On the real side, it's the actual
  formal uncertainty from a least-squares orbit fit against real
  astrometry. Both land in the same schema columns, but if the synthetic
  heuristic's uncertainty distribution doesn't resemble how real
  orbit-fit uncertainty actually behaves, the network could be
  miscalibrated on real data in a way training metrics wouldn't reveal.
- **Reference frame consistency is assumed, not checked.** Both sides use
  heliocentric, ecliptic, J2000 elements by convention (matching the JPL
  SSD table `GIANT_PLANETS` was corrected against, and JPL SBDB's default
  output) — nothing in code verifies a future data source uses the same
  frame.
- **Per-object epoch consistency in real data is assumed, not checked.**
  Each SBDB response carries its own orbit-fit epoch (e.g.
  `"epoch": "2461200.5"`), but `obsdata.fetch.RawOrbit` discards it —
  objects pulled into one x are treated as simultaneous without verifying
  their individual fit epochs are actually close together.

## Architecture

```
[prior.yaml] → [simgen.priors: sample θ, nuisance] → [simgen.worker: REBOUND integration]
                                                                │  (single final-epoch snapshot)
                                                                ▼
                                              [simgen.selection: survey forward model]
                                                                │
                                                                ▼  featurelib.build_feature_set
                                              [simgen.orchestrate: sharded Parquet dataset]
                                                                │
                                                                ▼
                                                   [model.train: NPE training loop]
                                                                │
                                                                ▼
                                                [model.posterior_net: q_φ(θ|x), trained]
                                                                ▲
                                                                │  featurelib.build_feature_set (same code)
[real catalogs: JPL SBDB / OSSOS / DES / Rubin] → [obsdata.fetch, obsdata.orbitfit, obsdata.build_x] → x'_obs
```

### Model: why a transformer, and where it attends

`planetx.model.encoder.SetTransformerEncoder` runs self-attention **across
the set of TNOs**, not across time — time was already collapsed to one
epoch upstream. It deliberately omits positional encoding (object order is
meaningless), pads variable-size populations with a mask, and includes a
persistent "null object" token so attention stays well-defined even for an
empty detected set. A learned pooling-by-multihead-attention (PMA) head
turns the variable-size set into a fixed-size population embedding `z`,
which conditions `planetx.model.flow.ConditionalPosteriorFlow` (a zuko
Neural Spline Flow) to produce `q_φ(θ | z)`. Training minimizes
`-log q_φ(θ_true | z)` (`planetx.model.posterior_net.PosteriorNet.loss`) —
this NLL objective, not the architecture, is what gives a calibrated
posterior instead of a point estimate that would collapse multimodal
solutions to their mean.

### Network input/output shapes

Using the default construction —
`PosteriorNet(object_feature_dim=11, theta_dim=7, survey_meta_dim=3, d_model=64, n_layers=3, n_heads=4, n_seeds=4)`,
where `11 = len(OBJECT_FEATURE_KEYS)` and `7 = len(THETA_KEYS)` (`constants.py`):

**Inputs** (from `model/train.py`'s `collate`, or `cli.py`'s `model_infer` for one real population):

| tensor | shape | meaning |
|---|---|---|
| `features` | `[B, N, 11]` | padded population: `OBJECT_FEATURE_KEYS` columns per object |
| `key_padding_mask` | `[B, N]` (bool) | `True` = padding, ignore this slot |
| `survey_meta` | `[B, 3]` | `(sky_fraction, limiting_mag, tracking_efficiency)` |
| `theta` | `[B, 7]` | training label only (`THETA_KEYS` order) — absent at inference |

**Forward pass**, batch size `B`, population size `N`:

| stage | operation | shape |
|---|---|---|
| `ObjectEncoder` | per-object MLP | `[B, N, 11]` → `[B, N, 64]` |
| null-token concat | prepend learned, unmasked token | → `[B, N+1, 64]` |
| `SelfAttentionBlock` ×3 | self-attention across objects, no positional encoding | `[B, N+1, 64]` → `[B, N+1, 64]` |
| `PoolingByMultiheadAttention` | 4 seed vectors attend over the set | → `[B, 4, 64]` → reshaped `[B, 256]` |
| `+ survey_meta_encoder` | `Linear(3, 256)` added in | `z: [B, 256]` |
| `ConditionalPosteriorFlow` | zuko NSF conditioned on `z` | density over `θ ∈ ℝ⁷` |

**Outputs:**

| call | shape | meaning |
|---|---|---|
| `PosteriorNet.loss(...)` | scalar | `-mean(log q_φ(θ_true \| z))` — the training objective |
| `PosteriorNet.posterior_samples(..., n=n)` | `[n, B, 7]` | `n` samples from `q_φ(θ \| z)` — the actual inference deliverable, not a point estimate |

## Project layout

```
configs/prior.yaml         prior over θ (HPX) and nuisance parameters
src/planetx/
  constants.py              fixed physical constants, theta/feature key order
  config.py                 PriorConfig / Distribution, YAML loading
  simgen/
    priors.py               sample θ, nuisance from PriorConfig
    worker.py                one REBOUND simulation -> final-epoch state
    selection.py             survey selection forward model (x from raw state)
    orchestrate.py            parallel fan-out -> sharded Parquet dataset
  featurelib/
    features.py               build_feature_set(): SHARED by simgen and obsdata
  model/
    encoder.py                 SetTransformerEncoder (no positional encoding)
    flow.py                     ConditionalPosteriorFlow (zuko NSF)
    posterior_net.py             PosteriorNet: embed + NPE NLL loss
    train.py                      training loop over sharded Parquet
  obsdata/
    fetch.py                    JPL SBDB connector (functional) + OSSOS/DES/Rubin stubs
    orbitfit.py                  orbit + uncertainty extraction, refit fallback stub
    build_x.py                    real catalog -> x, via the SAME featurelib call
  cli.py                       `planetx` command group
tests/                       pytest suite (torch/zuko/rebound tests self-skip if absent)
```

## Setup (uv)

```bash
# install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync              # creates .venv, installs rebound/torch/zuko/etc. (reboundx is optional -- see below)
uv run pytest        # torch/zuko/rebound-dependent tests self-skip if a dep failed to build
```

**Intel Mac note:** torch stopped publishing macOS x86_64 wheels after 2.2.2
(Jan 2024) and doesn't support Python 3.13 until well after that. This repo
pins `torch>=2.2,<2.3`, caps `requires-python` at `<3.13`, and sets
`tool.uv.required-environments` so `uv sync` resolves a combination that
actually has wheels on Intel Macs. If you already ran `uv sync` before this
pin was added and it picked up a 3.13 interpreter, delete `.venv` and
`uv.lock` and re-run `uv sync`. On Apple Silicon or Linux you can drop both
constraints once Intel Mac support doesn't matter to you.

**REBOUND 5.1.0+ is broken on macOS (verified upstream bug, not this
repo's packaging):** 5.1.0 rewrote the WHFast512 integrator in x86
assembly. Its `setup.py`'s `is_x86_64()` check hard-codes `return False`
for `sys.platform == "darwin"` — i.e. for every Mac — so that assembly
object never gets built or linked there, but the C sources still reference
its symbols unconditionally. Result: `import rebound` fails with `dlopen
... symbol not found in flat namespace` for a `whfast512` symbol, on any
Mac, regardless of reboundx. Confirmed by diffing `setup.py` across
versions: 5.0.1 has no such assembly step (AVX512 there is a harmless
opt-in compile flag, off by default); 5.1.0/5.1.1 do. This project only
ever uses plain `whfast`, never `whfast512`, so `pyproject.toml` pins
`rebound>=4.4,<5.1.0` to stay on the last known-good line. If a future
rebound release fixes the darwin build, this pin can be relaxed.

**`reboundx` is an optional extra, not a base dependency:** it's only used
for the optional 1-PN GR correction (`simgen/worker.py`'s `_maybe_add_gr`,
gated behind `configs/prior.yaml`'s `use_gr: false` default) and is imported
lazily, so the default `uv sync` never needs to build it. If you want GR:

```bash
uv sync --extra gr
```

`tool.uv.no-build-isolation-package` and `tool.uv.extra-build-dependencies`
in `pyproject.toml` work around two build-only quirks unrelated to the
rebound-5.1.x bug above: reboundx's build step needs `rebound` already
installed in the *same* environment it builds in (hence build isolation is
disabled for it), and needs `setuptools` present there explicitly (its own
`build-system.requires` doesn't declare it). If `uv sync --extra gr` still
fails after that, it's worth filing against `reboundx` directly, since the
rest of this project doesn't depend on it.

`rebound` compiles a C extension on install; `torch` is a large download. If
`uv sync` is slow or fails in a constrained environment, run it
somewhere with normal network/compiler access — nothing else in this repo
depends on network access except `planetx.obsdata.fetch` (a real call to
the public JPL SBDB API).

## Usage

```bash
# 1. generate a (small, for a first smoke test) training set
uv run planetx simgen run --prior configs/prior.yaml --out data/train \
    --n-sims 200 --shard-size 50 --workers 4

# 2. train the posterior network
uv run planetx model train --data-dir data/train --out models/posterior_net.pt \
    --epochs 20

# 3. pull real elements for a designation list into network-ready x
#    (replace the placeholder survey-metadata flags with a real survey's
#    characterization before trusting the result -- see "Selection function" above)
uv run planetx obsdata build-x --designations "2015 RR245,2013 FT28" \
    --out data/x_obs.npz --sky-fraction 0.03 --limiting-mag 24.5

# 4. run inference: posterior samples over HPX's parameters
uv run planetx model infer --checkpoint models/posterior_net.pt \
    --x data/x_obs.npz --n-samples 5000 --out data/theta_posterior.npy
```

## What's real vs. stubbed in this scaffold

**Functional:** prior sampling, the full REBOUND simulation worker (giant
planets + HPX + primordial test-particle disk, WHFast integration with
optional IAS15+`gr_full` general-relativity correction), the Set
Transformer + normalizing-flow NPE model end to end, JPL SBDB queries
(verified against a live API call), the sharded-dataset training loop, and
the CLI wiring all of it together.

**Deliberately stubbed** (each raises `NotImplementedError` with a specific
TODO, rather than silently faking data):

- `simgen.selection.SimpleSelectionFunction` — an illustrative
  magnitude/sky-fraction/tracking-efficiency cut, **not** a real survey
  simulator. Swap in the [OSSOS Survey
  Simulator](https://github.com/OSSOS/SurveySimulator) or
  [Sorcha](https://github.com/dirac-institute/sorcha) before trusting
  results on real data — see "The selection function" above.
- `obsdata.fetch.OSSOSConnector` / `DESConnector` / `RubinConnector` — need
  wiring to each survey's actual public data release and selection-function
  characterization.
- `obsdata.orbitfit.refit_from_astrometry` — fallback orbit fit for objects
  without a usable SBDB covariance; needs an actual orbit-determination
  package (OpenOrb, find_orb) fed with archival/precovery astrometry.
- `GIANT_PLANETS` in `constants.py` uses J2000.0 elements from JPL SSD's
  reference table (verified — see "Fixed constants vs. nuisance parameters
  vs. θ" above for the correction history), not a live ephemeris. For
  production runs at a different epoch, replace it with a live JPL
  Horizons / DE440 query instead.
- θ/feature standardization before the flow (currently raw physical units —
  AU, degrees, Earth masses) is omitted; see the note in `model/train.py`.

## Validation checklist before trusting a real-data posterior

None of this scaffold's output should be treated as a scientific claim
until these pass — see the design discussion this project came out of:

1. **Simulation-based calibration (SBC) / coverage tests** on held-out
   simulations.
2. **Literature-benchmark recovery**: does the trained network recover
   published constraints (e.g. Batygin/Brown-style ~5–10 M⊕, a ~ 400–800 AU)
   when given a matching synthetic population?
3. **Null test**: run the known Solar System (no HPX) through the same
   selection function; the network should not hallucinate a planet.
4. **Cross-survey consistency**: run OSSOS-only, DES-only, and combined
   real catalogs through the trained network independently; disagreement
   between them is a red flag for a selection-function mismatch or a
   spurious signal, echoing the real Brown/Batygin-vs-OSSOS debate.
# xhunter
