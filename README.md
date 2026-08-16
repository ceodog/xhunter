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

### Fixed constants vs. nuisance parameters vs. θ

Three different roles, easy to conflate:

| | Sampled per simulation? | Marginalized? | Example |
|---|---|---|---|
| θ | yes | no — this is the label | HPX mass, a, e, i, Ω, ω, M |
| nuisance | yes | yes (never a label) | primordial disk mass/extent, disk excitation |
| fixed | no — same every run | n/a | known giant planet masses |

The giant planets' masses are known to far higher precision (spacecraft
tracking, DE440/DE441) than would meaningfully affect the HPX signal, so
they're held fixed (`planetx.constants.GIANT_PLANETS`) rather than sampled
as a nuisance latent — sampling them would waste model capacity for no
statistical benefit. Primordial disk properties, by contrast, are
genuinely uncertain and could confound HPX's effect on the TNO population
if not marginalized, so they *are* sampled every run and never fed back as
a label.

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

### Feature parity

The exact same feature-engineering code
(`planetx.featurelib.build_feature_set`) is called from both the
simulation pipeline (`planetx.simgen.selection`) and the real-data
pipeline (`planetx.obsdata.build_x`). This is what prevents train/inference
skew — if the two sides ever computed features differently, the network's
posterior on real data would be unreliable regardless of how well training
converged.

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
- `GIANT_PLANETS` in `constants.py` uses approximate mean J2000 elements;
  replace with a JPL Horizons / DE440 query for production runs.
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
