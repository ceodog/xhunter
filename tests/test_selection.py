import numpy as np

from planetx.simgen.selection import SimpleSelectionFunction, _heliocentric_distance, _solve_kepler


def test_solve_kepler_circular_orbit_matches_mean_anomaly():
    E = _solve_kepler(np.array([45.0, 200.0]), np.array([0.0, 0.0]))
    assert np.allclose(E, np.radians([45.0, 200.0]), atol=1e-6)


def test_heliocentric_distance_at_perihelion_and_aphelion():
    a, e = 100.0, 0.5
    r_peri = _heliocentric_distance(a, e, 0.0)  # M=0 -> perihelion
    r_apo = _heliocentric_distance(a, e, 180.0)  # M=180 -> aphelion
    assert np.isclose(r_peri, a * (1 - e), atol=1e-3)
    assert np.isclose(r_apo, a * (1 + e), atol=1e-3)


def test_selection_function_never_exceeds_input_population():
    rng = np.random.default_rng(0)
    tnos = [
        {"a": 400.0, "e": 0.4, "i": 20.0, "Omega": 30.0, "omega": 60.0, "M": rng.uniform(0, 360)}
        for _ in range(500)
    ]
    sel = SimpleSelectionFunction(sky_fraction=0.05, limiting_mag=24.5, tracking_efficiency=0.8)
    fset = sel.apply(tnos, rng)
    assert 0 <= fset.n_objects <= len(tnos)


def test_selection_function_full_sky_bright_objects_detects_most():
    rng = np.random.default_rng(1)
    tnos = [
        {"a": 60.0, "e": 0.05, "i": 2.0, "Omega": 0.0, "omega": 0.0, "M": rng.uniform(0, 360)}
        for _ in range(200)
    ]
    sel = SimpleSelectionFunction(
        sky_fraction=1.0, limiting_mag=30.0, tracking_efficiency=1.0,
        absolute_mag_range=(4.0, 5.0),
    )
    fset = sel.apply(tnos, rng)
    assert fset.n_objects == len(tnos)
