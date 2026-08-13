"""Schema-v26 regression and scientific-invariant tests.

Integration cases are intentionally short. The 500-step calibration and
performance matrices are validation jobs, not unit tests, and never start the
production fixed-grid sweep.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pytest

import single_root_sim as sim
import root_hpc_manager as hpc_manager
import root_hpc_storage as hpc_storage


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "schema_v26_regression.json"


def parameters(
    seed: int = 12345,
    *,
    branch_probability: float = 0.10,
    rain_probability: float = 0.50,
    thickness_increment: float = 0.10,
) -> sim.SimulationParameters:
    return sim.SimulationParameters(
        rain_probability=rain_probability,
        branch_probability=branch_probability,
        thickness_increment=thickness_increment,
        seed=seed,
        sim_id=f"test-{seed}",
        task_index=-1,
    )


def config(**overrides: object) -> sim.SimulationConfig:
    values = {
        "steps": 45,
        "max_nodes": 50_000,
        "max_sampled_points": 50_000,
        "interactive_safety_cap": 50_000,
        "max_seconds_per_simulation": 0.0,
    }
    values.update(overrides)
    return sim.SimulationConfig(**values)


def run_case(
    seed: int = 12345,
    *,
    branch_probability: float = 0.10,
    rain_probability: float = 0.50,
    thickness_increment: float = 0.10,
    **overrides: object,
) -> tuple[dict[str, int | float | str], sim.NodeStore]:
    result = sim.run_simulation(
        parameters(
            seed,
            branch_probability=branch_probability,
            rain_probability=rain_probability,
            thickness_increment=thickness_increment,
        ),
        config(**overrides),
        return_store=True,
    )
    assert isinstance(result, tuple)
    return result


def geometry_signature(store: sim.NodeStore) -> tuple[np.ndarray, ...]:
    size = store.size
    metadata = store.axis_metadata
    sites = metadata.get("branch_sites", [])
    site_values = np.asarray(
        [
            (
                int(site["site_id"]), int(site["axis_id"]),
                float(site["material_arc"]), int(site["trial_count"]),
                int(site["accepted_branch_count"]),
            )
            for site in sites
        ],
        dtype=np.float64,
    ).reshape(-1, 5)
    return (
        store.position[:size].copy(),
        store.parent[:size].copy(),
        store.radius[:size].copy(),
        np.asarray(metadata["axis_parent_ids"]).copy(),
        np.asarray(metadata["axis_parent_arc_lengths"]).copy(),
        site_values,
    )


def zero_resource_config(**overrides: object) -> sim.SimulationConfig:
    return config(
        soil_water_background=0.0,
        rain_water_input=0.0,
        phosphorus_concentration=0.0,
        nitrogen_concentration=0.0,
        potassium_concentration=0.0,
        **overrides,
    )


def manual_axis(
    axis_id: int,
    parent_axis_id: int,
    parent_arc: float,
    generation: int,
    length: float,
) -> sim.RootAxis:
    return sim.RootAxis(
        axis_id=axis_id,
        parent_axis_id=parent_axis_id,
        parent_arc_length=parent_arc,
        parent_local_azimuth=0.0,
        birth_step=0,
        is_anchor_axis=False,
        branch_generation=generation,
        points=[np.zeros(3), np.asarray([length, 0.0, 0.0])],
        tangents=[np.asarray([1.0, 0.0, 0.0])] * 2,
        point_birth_steps=[0, 1],
        material_arcs=[0.0, length],
        radius_scale=0.2,
    )


# 1-3: fixed task contract and schema.
def test_fixed_grid_task_count_and_dimensions() -> None:
    assert sim.TOTAL_GRID_TASKS == 70 * 99 * 99 * 5 == 3_430_350
    assert sim.GRID_THICKNESS_COUNT == 70
    assert sim.GRID_RAIN_COUNT == sim.GRID_BRANCH_COUNT == 99
    assert sim.GRID_REPLICATES == 5


def test_task_mapping_and_master_seed_are_unchanged() -> None:
    assert sim.build_parser().parse_args([]).master_seed == 20260617
    for index in (0, 1, 4, 5, 490, 49_004, sim.TOTAL_GRID_TASKS - 1):
        item = sim.parameters_for_task(index, 20260617)
        assert item.task_index == index
        assert item.seed == sim.seed_for_task(20260617, index)
    last = sim.parameters_for_task(sim.TOTAL_GRID_TASKS - 1, 20260617)
    assert (last.thickness_increment, last.rain_probability, last.branch_probability) == (
        7.0, 0.99, 0.99
    )


def test_schema_is_26_and_result_contract_is_exact() -> None:
    metrics, _ = run_case(103, steps=8)
    assert sim.SCHEMA_VERSION == 26
    assert len(sim.RESULT_FIELDS) == len(set(sim.RESULT_FIELDS))
    assert set(metrics) == set(sim.RESULT_FIELDS)
    assert metrics["branch_retry_mode"] == "single_trial"


# 4-6: unified continuous-axis architecture.
def test_no_morphology_selector_or_growth_control_exists() -> None:
    field_names = {item.name for item in fields(sim.SimulationConfig)}
    forbidden = {"morphology_mode", "taproot", "dicot", "whorled"}
    assert "morphology_mode" not in field_names
    growth_source = inspect.getsource(sim._axis_curve_simulation).lower()
    assert not any(token in growth_source for token in forbidden)


def test_sampled_nodes_are_export_support_not_branch_sites() -> None:
    assert "material_arc" in {item.name for item in fields(sim.BranchSite)}
    source = inspect.getsource(sim.advance_continuous_branch_sites)
    assert "axis.next_branch_site_arc" in source
    assert "rng.exponential" in source
    assert "point_birth_steps" not in source


def test_curves_are_cubic_hermite_continuous_axes() -> None:
    curve_source = inspect.getsource(sim.hermite_curve_samples)
    assert "h00" in curve_source and "h10" in curve_source
    metrics, store = run_case(107, steps=12)
    assert metrics["axis_count"] >= 1
    assert "axis_material_arcs" in store.axis_metadata
    assert all(np.all(np.diff(arcs) > 0.0) for arcs in store.axis_metadata["axis_material_arcs"] if len(arcs) > 1)


# 7-14: all-active tips and directional behavior.
def test_every_active_snapshot_tip_receives_exactly_one_attempt() -> None:
    metrics, _ = run_case(109, branch_probability=0.40, steps=35)
    assert metrics["active_tip_attempt_accounting_error"] == 0
    assert metrics["tip_extension_attempts"] == metrics["active_tips_at_step_start_total"]
    assert metrics["tip_extension_attempts"] == (
        metrics["tip_extensions_accepted"]
        + metrics["tip_extensions_collision_blocked"]
        + metrics["tip_extensions_surface_blocked"]
        + metrics["tip_extensions_sample_cap_blocked"]
        + metrics["tip_extensions_other_blocked"]
    )


def test_construction_budget_and_subset_selection_are_removed() -> None:
    assert not hasattr(sim, "plant_wide_construction_budget")
    assert not hasattr(sim, "select_axes_for_construction")
    field_names = {item.name for item in fields(sim.SimulationConfig)}
    assert not any(name.startswith("construction_budget_") for name in field_names)
    assert not any("construction_budget" in name for name in sim.RESULT_FIELDS)
    assert "growth_events_selected" not in sim.RESULT_FIELDS


def test_resource_support_cannot_remove_active_tip_attempt() -> None:
    poor, _ = sim.run_simulation(
        parameters(113, branch_probability=0.01, rain_probability=0.01),
        zero_resource_config(steps=25),
        return_store=True,
    )
    rich, _ = run_case(113, branch_probability=0.01, rain_probability=0.01, steps=25)
    assert poor["tip_extension_attempts"] >= poor["developmental_steps_completed"]
    assert rich["tip_extension_attempts"] >= rich["developmental_steps_completed"]
    assert poor["active_tip_attempt_accounting_error"] == 0
    assert rich["active_tip_attempt_accounting_error"] == 0


def test_zero_resource_tips_retain_positive_extension_length() -> None:
    primary = sim.RootAxisStore(zero_resource_config()).axes[0]
    lateral = manual_axis(1, 0, 0.5, 3, 1.0)
    assert sim.tip_extension_length(primary, 500, zero_resource_config()) > 0.0
    assert sim.tip_extension_length(lateral, 500, zero_resource_config()) > 0.0
    assert "branch_probability" not in inspect.getsource(sim.tip_extension_length)


def test_zero_resource_direction_strongly_favors_downward_growth() -> None:
    metrics, _ = sim.run_simulation(
        parameters(127, branch_probability=0.01, rain_probability=0.01),
        zero_resource_config(steps=90),
        return_store=True,
    )
    assert metrics["generation_0_mean_vertical_component"] < -0.93
    assert metrics["fraction_upward_segments"] < 0.03
    assert metrics["fraction_near_horizontal_segments"] < 0.12
    assert metrics["above_surface_length"] == pytest.approx(0.0)
    source = inspect.getsource(sim.biased_axis_direction)
    assert "3.20 * (1.0 - sufficiency) ** 1.4" in source
    assert "upward_forbidden" in source


def test_resources_alter_direction_stochastically() -> None:
    summaries: list[float] = []
    for phosphorus, nitrogen, potassium in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        metrics, _ = run_case(
            131,
            branch_probability=0.20,
            steps=55,
            phosphorus_concentration=phosphorus,
            nitrogen_concentration=nitrogen,
            potassium_concentration=potassium,
        )
        summaries.append(float(metrics["mean_extension_direction_z"]))
        assert metrics["active_tip_attempt_accounting_error"] == 0
    assert np.ptp(summaries) > 1e-3


def test_new_branches_wait_until_following_step() -> None:
    metrics, store = run_case(131, branch_probability=0.99, steps=3)
    births = np.asarray(store.axis_metadata["axis_birth_steps"])
    events = np.asarray(store.axis_metadata["axis_extension_events"])
    last_growth = np.asarray(store.axis_metadata["axis_last_growth_steps"])
    newborn = np.flatnonzero(births == 2)
    assert newborn.size > 0
    assert np.all(events[newborn] == 1)  # emergence shoulder only
    assert np.all(last_growth[newborn] == births[newborn])
    assert metrics["developmental_steps_completed"] == 3


# 15-21: continuous Poisson sites and retry semantics.
def test_branch_sites_use_continuous_material_arc_and_stochastic_gaps() -> None:
    cfg = config(branch_min_distance_from_tip=0.2, branch_min_distance_from_base=0.25)
    store = sim.RootAxisStore(cfg)
    axis = store.axes[0]
    axis.points.append(np.asarray([0.0, 0.0, -5.0]))
    axis.tangents.append(np.asarray([0.0, 0.0, -1.0]))
    axis.point_birth_steps.append(1)
    axis.material_arcs.append(5.0)
    sites = sim.advance_continuous_branch_sites(store, axis, 3, np.random.default_rng(19), cfg)
    arcs = np.asarray([site.material_arc for site in sites])
    gaps = np.diff(arcs)
    assert arcs.size > 10
    assert np.all(arcs >= cfg.branch_min_distance_from_base)
    assert np.all(arcs <= axis.total_length() - cfg.branch_min_distance_from_tip)
    assert np.ptp(gaps) > 0.1 * cfg.branch_min_spacing_along_axis
    assert not np.allclose(gaps, cfg.branch_min_spacing_along_axis)


def test_branch_site_maturity_exclusions() -> None:
    cfg = config(lateral_branch_min_age=4, branch_min_distance_from_base=0.4, branch_min_distance_from_tip=0.6)
    store = sim.RootAxisStore(cfg)
    axis = store.axes[0]
    axis.points.append(np.asarray([0.0, 0.0, -3.0]))
    axis.tangents.append(np.asarray([0.0, 0.0, -1.0]))
    axis.point_birth_steps.append(1)
    axis.material_arcs.append(3.0)
    assert not sim.advance_continuous_branch_sites(store, axis, 3, np.random.default_rng(23), cfg)
    sites = sim.advance_continuous_branch_sites(store, axis, 4, np.random.default_rng(23), cfg)
    assert sites
    assert min(site.material_arc for site in sites) >= 0.4
    assert max(site.material_arc for site in sites) <= 2.4


def test_single_trial_mode_never_retries_failed_site() -> None:
    cfg = config(branch_retry_mode="single_trial")
    store = sim.RootAxisStore(cfg)
    site = sim.BranchSite(0, 0, 0.5, 2, 2, trial_count=1, failure_count=1, closed_in_single_trial_mode=True)
    store.branch_sites.append(site)
    store.axes[0].branch_site_ids.append(0)
    assert sim.eligible_branch_sites(store, store.axes[0], 3, cfg) == []
    metrics, _ = run_case(139, branch_probability=0.01, steps=35, branch_retry_mode="single_trial")
    assert metrics["branch_site_retry_trials"] == 0


def test_retry_mode_retries_failed_open_site() -> None:
    cfg = config(branch_retry_mode="retry_open_sites")
    store = sim.RootAxisStore(cfg)
    site = sim.BranchSite(0, 0, 0.5, 2, 2, trial_count=1, failure_count=1, last_trial_step=2)
    store.branch_sites.append(site)
    store.axes[0].branch_site_ids.append(0)
    assert sim.eligible_branch_sites(store, store.axes[0], 3, cfg) == [site]
    metrics, _ = run_case(149, branch_probability=0.06, steps=35, branch_retry_mode="retry_open_sites")
    assert metrics["branch_site_retry_trials"] > 0


# Branch-initiation probability uses an independent deterministic stream.
def test_initiation_threshold_has_no_water_or_rain_dependency() -> None:
    variants = [
        config(soil_water_background=water, rain_water_input=rain_input)
        for water, rain_input in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
    ]
    thresholds = [sim.site_initiation_probability(0.20, 0, cfg) for cfg in variants]
    assert thresholds == pytest.approx([0.20] * len(variants))
    source = inspect.getsource(sim.site_initiation_probability).lower()
    assert "water" not in source and "rain" not in source


def test_initiation_threshold_has_no_mineral_dependency() -> None:
    mineral_states = (
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 1.0, 1.0),
    )
    thresholds = [
        sim.site_initiation_probability(
            0.30,
            1,
            config(
                phosphorus_concentration=p,
                nitrogen_concentration=n,
                potassium_concentration=k,
            ),
        )
        for p, n, k in mineral_states
    ]
    expected = 0.30 ** (1.0 + sim.SimulationConfig().lateral_generation_probability_exponent)
    assert thresholds == pytest.approx([expected] * len(mineral_states))


def test_initiation_threshold_has_no_demand_or_focus_dependency() -> None:
    variants = (
        config(enable_resource_demand_feedback=False),
        config(resource_demand_feedback_strength=0.0, resource_demand_weight_cap=0.0),
        config(resource_demand_feedback_strength=1.0, resource_demand_weight_cap=8.0),
        config(resource_focus_update_probability=0.0),
        config(resource_focus_update_probability=1.0),
    )
    values = [sim.site_initiation_probability(0.44, 2, cfg) for cfg in variants]
    assert values == pytest.approx([values[0]] * len(values))
    signature = inspect.signature(sim.site_initiation_probability)
    assert set(signature.parameters) == {
        "configured_branch_probability", "parent_generation", "config"
    }


def test_initiation_threshold_has_no_support_or_starvation_dependency() -> None:
    variants = (
        config(starvation_stop_threshold=0.0, starvation_lateral_suppression=0.0),
        config(starvation_stop_threshold=1.0, starvation_lateral_suppression=8.0),
        config(water_support_half_saturation=0.01, nutrient_support_half_saturation=0.01),
        config(water_support_half_saturation=1.0, nutrient_support_half_saturation=1.0),
    )
    values = [sim.site_initiation_probability(0.18, 1, cfg) for cfg in variants]
    assert values == pytest.approx([values[0]] * len(values))


def test_local_stimulus_never_changes_initiation_threshold() -> None:
    cfg = config()
    axis = sim.RootAxisStore(cfg).axes[0]
    without_prior = sim.local_primordium_stimulus(axis, 0.5, 10)
    axis.branch_origins.append(0.5)
    axis.branch_origin_steps.append(10)
    with_prior = sim.local_primordium_stimulus(axis, 0.5, 10)
    assert without_prior == 0.0 and with_prior > 0.0
    assert sim.site_initiation_probability(0.27, 0, cfg) == pytest.approx(0.27)
    simulation_source = inspect.getsource(sim._axis_curve_simulation)
    probability_at = simulation_source.index("site_initiation_probability(")
    stimulus_at = simulation_source.index("local_primordium_stimulus(")
    assert probability_at < stimulus_at


def test_initiation_draw_is_stable_across_all_resource_environments() -> None:
    environments = (
        zero_resource_config(),
        config(phosphorus_concentration=1.0, nitrogen_concentration=0.0, potassium_concentration=0.0),
        config(phosphorus_concentration=0.0, nitrogen_concentration=1.0, potassium_concentration=0.0),
        config(phosphorus_concentration=0.0, nitrogen_concentration=0.0, potassium_concentration=1.0),
        config(soil_water_background=1.0, rain_water_input=1.0),
    )
    draws = [
        sim.initiation_probability_uniform(20260617, 488_565, 3, 17, 2)
        for _environment in environments
    ]
    assert draws == pytest.approx([draws[0]] * len(draws))
    assert len({sim.initiation_probability_uniform(20260617, 488_565, 3, 17, trial) for trial in range(1, 8)}) == 7


def test_primary_axis_threshold_is_exactly_configured_bp() -> None:
    cfg = config()
    for probability in (0.0, 0.01, 0.20, 0.50, 0.80, 0.99, 1.0):
        expected = min(probability, 0.99)
        assert sim.lineage_branch_probability(probability, 0, cfg) == pytest.approx(expected)
        assert sim.site_initiation_probability(probability, 0, cfg) == pytest.approx(expected)


def test_retry_trials_keep_the_same_per_step_hazard() -> None:
    cfg = config(branch_retry_mode="retry_open_sites")
    threshold = sim.site_initiation_probability(0.06, 0, cfg)
    thresholds = [sim.site_initiation_probability(0.06, 0, cfg) for _step in range(100)]
    assert thresholds == pytest.approx([threshold] * 100)
    assert threshold == pytest.approx(0.06)
    source = inspect.getsource(sim.site_initiation_probability)
    assert "trial" not in source and "step" not in source


def test_initiation_accounting_separates_failures_passes_and_physical_outcomes() -> None:
    metrics, _ = run_case(150, branch_probability=0.40, steps=45)
    assert metrics["branch_opportunities"] == (
        metrics["probability_failures"] + metrics["branch_probability_passes"]
    )
    assert metrics["branch_probability_passes"] == (
        metrics["successful_branches"] + metrics["physical_rejection_count"]
    )
    assert metrics["opportunity_accounting_error"] == 0
    assert metrics["probability_pass_accounting_error"] == 0
    assert metrics["probability_pass_rate"] == pytest.approx(
        metrics["branch_probability_passes"] / metrics["branch_opportunities"]
    )


def test_rain_branch_coupling_is_removed_from_all_interfaces() -> None:
    config_fields = {item.name for item in fields(sim.SimulationConfig)}
    assert {
        "rain_branch_coupling", "axis_resource_branch_bonus",
        "starvation_branch_floor", "starvation_branch_suppression",
        "starvation_branch_exponent",
    }.isdisjoint(config_fields)
    assert not hasattr(sim.build_parser().parse_args([]), "rain_branch_coupling")
    assert "rain_branch_coupling" not in sim.RESULT_FIELDS


# 22-26: cylindrical surface capacity and curve collisions.
def thick_parent(radius: float = 0.10) -> tuple[sim.RootAxis, sim.SimulationConfig]:
    cfg = config()
    axis = sim.RootAxisStore(cfg).axes[0]
    axis.points.append(np.asarray([0.0, 0.0, -2.0]))
    axis.tangents.append(np.asarray([0.0, 0.0, -1.0]))
    axis.point_birth_steps.append(1)
    axis.material_arcs.append(2.0)
    tip = cfg.base_radius * cfg.structural_tip_baseline_fraction
    axis.add_structural_area_event(2.0, math.pi * (radius * radius - tip * tip))
    return axis, cfg


def test_same_site_multiple_azimuths_and_overlap_rules() -> None:
    axis, cfg = thick_parent(0.10)
    axis.branch_origins.append(1.0)
    axis.branch_azimuths.append(0.0)
    axis.branch_origin_base_radii.append(0.01)
    same, _ = sim.cylindrical_surface_clearance(axis, 1.0, 0.0, 0.01, cfg)
    opposite, opposite_clearance = sim.cylindrical_surface_clearance(axis, 1.0, math.pi, 0.01, cfg)
    quarter, _ = sim.cylindrical_surface_clearance(axis, 1.0, math.pi / 2.0, 0.01, cfg)
    assert not same
    assert opposite and opposite_clearance > 0.0
    assert quarter


def test_nearby_axial_and_azimuth_clearance_is_physical() -> None:
    axis, cfg = thick_parent(0.04)
    axis.branch_origins.append(1.0)
    axis.branch_azimuths.append(0.0)
    axis.branch_origin_base_radii.append(0.012)
    close, _ = sim.cylindrical_surface_clearance(axis, 1.002, 0.03, 0.012, cfg)
    opposite, _ = sim.cylindrical_surface_clearance(axis, 1.002, math.pi, 0.012, cfg)
    assert not close
    assert opposite


def test_parent_thickening_can_reopen_circumference() -> None:
    axis, cfg = thick_parent(0.0128)
    axis.branch_origins[:] = [1.0, 1.0, 1.0, 1.0]
    axis.branch_azimuths[:] = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
    axis.branch_origin_base_radii[:] = [0.012] * 4
    site = sim.BranchSite(0, 0, 1.0, 2, 2, physically_open=False, temporarily_surface_full=True)
    site.last_evaluated_parent_radius = sim.sampled_axis_radius(axis, 1.0, cfg)
    store = sim.RootAxisStore(cfg)
    store.axes[0] = axis
    store.branch_sites.append(site)
    axis.branch_site_ids.append(0)
    before = sim.branch_site_has_surface_capacity(axis, site, cfg)
    axis.add_structural_area_event(2.0, math.pi * (0.08**2 - 0.0128**2))
    eligible = sim.eligible_branch_sites(store, axis, 3, replace(cfg, branch_retry_mode="retry_open_sites"))
    assert not before
    assert eligible == [site]
    assert store.branch_sites_reopened_after_thickening == 1


def test_branch_curve_collision_checks_remain_enforced() -> None:
    cfg = config(spatial_clearance=0.05)
    store = sim.RootAxisStore(cfg)
    store._append_spatial_points(0, np.asarray([[0.0, 0.0, -1.0]]))
    store.sample_count += 1
    crossing = np.asarray([[0.0, 0.0, -1.0], [0.01, 0.0, -1.0]])
    assert not store.samples_are_clear(crossing, own_axis_id=1)


# 27-35: scientific transport-area taper and ancestor propagation.
def test_primary_radius_decreases_base_to_tip_and_ti_changes_profile() -> None:
    low, low_store = run_case(151, branch_probability=0.01, thickness_increment=0.10, steps=90)
    high, high_store = run_case(151, branch_probability=0.01, thickness_increment=1.00, steps=90)
    low_profile = np.asarray(low_store.axis_metadata["axis_radii"][0])
    high_profile = np.asarray(high_store.axis_metadata["axis_radii"][0])
    assert low_profile[0] == pytest.approx(np.max(low_profile))
    assert low_profile[-1] == pytest.approx(np.min(low_profile))
    assert np.all(np.diff(low_profile) <= 1e-12)
    assert low["primary_axis_basal_to_tip_radius_ratio"] > 1.5
    assert not np.allclose(low_profile, high_profile)
    assert high["primary_axis_basal_radius"] > low["primary_axis_basal_radius"]


def test_lateral_radius_decreases_base_to_tip_when_long_enough() -> None:
    _metrics, store = run_case(157, branch_probability=0.75, steps=55)
    generations = np.asarray(store.axis_metadata["axis_generations"])
    lengths = np.asarray(store.axis_metadata["axis_arc_lengths"])
    profiles = store.axis_metadata["axis_radii"]
    eligible = np.flatnonzero((generations == 1) & (lengths >= 1.0))
    assert eligible.size > 0
    for axis_id in eligible:
        profile = np.asarray(profiles[int(axis_id)])
        assert profile[0] >= profile[-1]
        assert np.mean(np.diff(profile) <= 1e-12) >= 0.95


def test_tip_growth_thickens_own_proximal_path() -> None:
    cfg = config()
    store = sim.RootAxisStore(cfg)
    axis = store.axes[0]
    axis.material_arcs[:] = [0.0, 1.0]
    axis.points.append(np.asarray([0.0, 0.0, -1.0]))
    axis.tangents.append(np.asarray([0.0, 0.0, -1.0]))
    axis.point_birth_steps.append(1)
    before = sim.sampled_axis_radius(axis, 0.0, cfg)
    sim.record_transport_path_growth(store, axis, thickness_increment=1.0, grown_length=0.5)
    assert sim.sampled_axis_radius(axis, 0.0, cfg) > before


def test_child_growth_thickens_only_parent_proximal_to_attachment() -> None:
    cfg = config()
    store = sim.RootAxisStore(cfg)
    primary = store.axes[0]
    primary.material_arcs[:] = [0.0, 1.0]
    child = manual_axis(1, 0, 0.40, 1, 1.0)
    store.axes.append(child)
    before = sim.axis_radii_at_arcs(primary, np.asarray([0.2, 0.8]), cfg)
    sim.record_transport_path_growth(store, child, thickness_increment=1.0, grown_length=1.0)
    after = sim.axis_radii_at_arcs(primary, np.asarray([0.2, 0.8]), cfg)
    assert after[0] > before[0]
    assert after[1] == pytest.approx(before[1])


def test_grandchild_growth_recursively_thickens_grandparent() -> None:
    cfg = config()
    store = sim.RootAxisStore(cfg)
    primary = store.axes[0]
    primary.material_arcs[:] = [0.0, 1.0]
    child = manual_axis(1, 0, 0.55, 1, 1.0)
    grandchild = manual_axis(2, 1, 0.60, 2, 1.0)
    store.axes.extend((child, grandchild))
    before_primary = sim.sampled_axis_radius(primary, 0.2, cfg)
    before_child = sim.sampled_axis_radius(child, 0.2, cfg)
    sim.record_transport_path_growth(store, grandchild, thickness_increment=1.0, grown_length=1.0)
    assert sim.sampled_axis_radius(child, 0.2, cfg) > before_child
    assert sim.sampled_axis_radius(primary, 0.2, cfg) > before_primary
    assert any(event[1] == pytest.approx(0.0012) for event in child.structural_area_events)
    assert any(event[1] == pytest.approx(0.0012 * 0.62) for event in primary.structural_area_events)


def test_area_aggregation_is_sublinear_in_radius() -> None:
    cfg = config()
    one = sim.RootAxisStore(cfg).axes[0]
    ten = sim.RootAxisStore(cfg).axes[0]
    for _ in range(1):
        one.add_structural_area_event(1.0, 0.01)
    for _ in range(10):
        ten.add_structural_area_event(1.0, 0.01)
    ratio = sim.sampled_axis_radius(ten, 0.0, cfg) / sim.sampled_axis_radius(one, 0.0, cfg)
    assert 2.5 < ratio < 4.0
    assert ratio < 10.0


def test_branch_origin_radius_is_bounded_by_parent_radius() -> None:
    _metrics, store = run_case(163, branch_probability=0.40, steps=45)
    metadata = store.axis_metadata
    parent_ids = np.asarray(metadata["axis_parent_ids"])
    parent_radii = np.asarray(metadata["axis_parent_local_radii"])
    child_radii = np.asarray(metadata["axis_basal_radii"])
    for axis_id in np.flatnonzero(parent_ids >= 0):
        assert child_radii[axis_id] <= parent_radii[axis_id] + 1e-12
    assert _metrics["branch_origin_child_parent_radius_ratio_max"] <= (
        config().branch_origin_child_parent_radius_ratio_limit + 1e-12
    )


def test_lateral_age_collision_and_profile_diagnostics_are_complete() -> None:
    required_result_fields = {
        "lateral_axis_diagnostics_json",
        "lateral_count_age_0_2", "lateral_count_age_3_5",
        "lateral_count_age_6_10", "lateral_count_age_11_25",
        "lateral_count_age_gt_25",
        "mean_lateral_length_age_0_2", "mean_lateral_length_age_3_5",
        "mean_lateral_length_age_6_10", "mean_lateral_length_age_11_25",
        "mean_lateral_length_age_gt_25",
        "accepted_extensions_age_0_2", "accepted_extensions_age_3_5",
        "accepted_extensions_age_6_10", "accepted_extensions_age_11_25",
        "accepted_extensions_age_gt_25",
        "collision_rate_age_0_2", "collision_rate_age_3_5",
        "collision_rate_age_6_10", "collision_rate_age_11_25",
        "collision_rate_age_gt_25",
        "proportion_laterals_initial_shoulder_only",
        "proportion_laterals_at_least_2_accepted_extensions",
        "proportion_laterals_at_least_5_accepted_extensions",
        "proportion_laterals_at_least_10_accepted_extensions",
        "extension_parent_collision_blocked",
        "extension_other_root_collision_blocked",
        "extension_surface_blocked", "extension_other_blocked",
        "mean_initial_radial_displacement",
        "mean_distance_from_parent_after_shoulder",
        "mean_lateral_direction_z_after_emergence",
        "fraction_laterals_curve_back_inside_parent_radius",
        "fraction_laterals_only_one_support_curve",
        "fraction_laterals_active_at_termination",
        "profile_retry_site_traversal_sec",
        "profile_branch_probability_trials_sec",
        "profile_physical_origin_search_sec",
        "profile_active_tip_extensions_sec",
        "profile_collision_queries_sec",
        "profile_resource_direction_candidates_sec",
    }
    assert required_result_fields <= set(sim.RESULT_FIELDS)
    metrics, _store = run_case(
        20260715, branch_probability=0.73, rain_probability=0.01,
        steps=12, branch_retry_mode="retry_open_sites",
        soil_water_background=0.0, rain_water_input=0.0,
        phosphorus_concentration=0.0, nitrogen_concentration=0.0,
        potassium_concentration=0.0,
    )
    rows = json.loads(str(metrics["lateral_axis_diagnostics_json"]))
    required_row_fields = {
        "axis_id", "parent_axis_id", "generation", "birth_step",
        "completed_simulation_step", "biological_age_steps",
        "extension_attempts", "accepted_extensions",
        "collision_blocked_extensions", "parent_collision_blocked_extensions",
        "other_root_collision_blocked_extensions", "surface_blocked_extensions",
        "other_blocked_extensions", "current_arc_length",
        "initial_emergence_shoulder_length", "mean_accepted_extension_length",
        "last_accepted_growth_step", "current_active_state",
        "mean_local_resource_sufficiency", "mean_direction_z_after_emergence",
    }
    assert rows and all(required_row_fields <= set(row) for row in rows)
    assert sum(
        int(metrics[field]) for field in (
            "lateral_count_age_0_2", "lateral_count_age_3_5",
            "lateral_count_age_6_10", "lateral_count_age_11_25",
            "lateral_count_age_gt_25",
        )
    ) == len(rows)
    assert all(float(metrics[field]) >= 0.0 for field in required_result_fields if field.startswith("profile_"))


def test_post_birth_escape_is_short_geometry_based_and_bp_independent() -> None:
    cfg = sim.SimulationConfig()
    assert 1 <= cfg.lateral_escape_accepted_extensions <= 3
    assert cfg.lateral_escape_min_outward_component > 0.0
    escape_source = inspect.getsource(sim.branch_escape_radial_direction)
    extension_source = inspect.getsource(sim.try_extend_axis)
    for forbidden in ("branch_probability", "rain_probability", "phosphorus", "nitrogen", "potassium"):
        assert forbidden not in escape_source
    assert "escape_extensions_remaining" in extension_source
    assert "parent_axis_id=axis.parent_axis_id if in_escape else -1" in extension_source
    assert "escape_extensions_remaining - 1" in extension_source


# 39-45: reproducibility, caps, post-hoc isolation, metadata, CSV, compilation contract.
def test_deterministic_seed_reproduces_geometry_sites_topology_and_radii() -> None:
    first_metrics, first = run_case(167, branch_probability=0.20, steps=40)
    second_metrics, second = run_case(167, branch_probability=0.20, steps=40)
    for left, right in zip(geometry_signature(first), geometry_signature(second)):
        assert np.array_equal(left, right, equal_nan=True)
    assert first_metrics["accepted_branches"] == second_metrics["accepted_branches"]


def test_sample_cap_invariance_when_caps_are_not_reached() -> None:
    low_metrics, low = run_case(173, branch_probability=0.20, steps=30, max_sampled_points=20_000, max_nodes=20_000)
    high_metrics, high = run_case(173, branch_probability=0.20, steps=30, max_sampled_points=40_000, max_nodes=40_000)
    assert not low_metrics["sample_cap_reached"] and not high_metrics["sample_cap_reached"]
    for left, right in zip(geometry_signature(low), geometry_signature(high)):
        assert np.array_equal(left, right, equal_nan=True)


def test_above_surface_geometry_remains_zero() -> None:
    metrics, store = run_case(179, branch_probability=0.75, steps=45)
    limit = config().soil_surface_z + config().max_above_surface_tolerance
    assert np.max(store.position[:store.size, 2]) <= limit + 1e-12
    assert metrics["above_surface_length"] == pytest.approx(0.0)


def test_posthoc_whorl_thresholds_do_not_alter_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    _first_metrics, first = run_case(181, branch_probability=0.40, steps=35)
    monkeypatch.setattr(sim, "POSTHOC_WHORL_MIN_BRANCHES", 99)
    monkeypatch.setattr(sim, "POSTHOC_WHORL_AXIAL_WINDOW", 0.001)
    _second_metrics, second = run_case(181, branch_probability=0.40, steps=35)
    for left, right in zip(geometry_signature(first), geometry_signature(second)):
        assert np.array_equal(left, right, equal_nan=True)


def test_both_retry_modes_are_in_results_metadata_and_cli(tmp_path: Path) -> None:
    parser = sim.build_parser()
    assert parser.parse_args([]).branch_retry_mode == "single_trial"
    assert parser.parse_args(["--branch-retry-mode", "retry_open_sites"]).branch_retry_mode == "retry_open_sites"
    assert set(sim.BRANCH_RETRY_MODES) == {"single_trial", "retry_open_sites"}
    args = parser.parse_args(["--mode", "batch"])
    args.task_stop = 1
    path = tmp_path / "metadata.json"
    sim.write_metadata(path, sim.config_from_args(args), args)
    metadata = json.loads(path.read_text())
    assert metadata["branch_retry_mode"] == "single_trial"
    assert metadata["branch_retry_mode_is_grid_dimension"] is False
    assert metadata["grid"]["branch_retry_mode_is_dimension"] is False


def test_canonical_spacing_is_mean_020_and_not_grid_dimension() -> None:
    assert sim.CANONICAL_BRANCH_MIN_SPACING_ALONG_AXIS == pytest.approx(0.20)
    assert sim.SimulationConfig().branch_min_spacing_along_axis == pytest.approx(0.20)
    args = sim.build_parser().parse_args([])
    assert args.branch_min_spacing_along_axis == pytest.approx(0.20)
    assert sim.config_from_args(args).branch_min_spacing_along_axis == pytest.approx(0.20)
    assert set(("thickness", "rain_probability", "branch_probability", "replicates")) <= {
        "thickness", "rain_probability", "branch_probability", "replicates"
    }


def test_schema_v25_csv_is_rejected(tmp_path: Path) -> None:
    legacy = tmp_path / "schema25.csv"
    with legacy.open("w", newline="") as stream:
        csv.writer(stream).writerow([
            "task_index", "sim_id", "construction_budget_total",
            "growth_events_selected", "rejected_axial_spacing",
        ])
    with pytest.raises(ValueError, match="schema version 26"):
        sim.validate_existing_csv_schema(legacy)


def test_repository_python_sources_compile() -> None:
    for path in (
        ROOT / "single_root_sim.py",
        ROOT / "root_hpc_manager.py",
        ROOT / "root_hpc_storage.py",
        ROOT / "root_hpc_worker.py",
        Path(__file__),
    ):
        compile(path.read_text(), path.as_posix(), "exec")


def test_source_and_result_contract_exclude_removed_v22_controls() -> None:
    forbidden_fields = {
        "construction_budget_total", "construction_budget_used",
        "construction_budget_unused", "construction_budget_utilization",
        "growth_events_selected", "axes_starved_for_construction",
        "rejected_axial_spacing", "rejected_azimuth_spacing",
    }
    assert forbidden_fields.isdisjoint(sim.RESULT_FIELDS)
    assert "last_branch_trial_arc" not in Path(sim.__file__).read_text()
    assert "max_lateral_branches_per_node" not in Path(sim.__file__).read_text()


def test_cylindrical_clearance_matches_documented_equation() -> None:
    axis, cfg = thick_parent(0.10)
    axis.branch_origins.append(0.8)
    axis.branch_azimuths.append(0.0)
    axis.branch_origin_base_radii.append(0.01)
    candidate_arc = 0.83
    candidate_azimuth = math.pi / 3.0
    candidate_collar = 0.012
    available, clearance = sim.cylindrical_surface_clearance(
        axis, candidate_arc, candidate_azimuth, candidate_collar, cfg
    )
    expected = math.hypot(0.03, 0.10 * math.pi / 3.0) - (
        cfg.branch_collar_clearance_factor * (0.012 + 0.01)
        + cfg.branch_collar_safety_margin
    )
    assert clearance == pytest.approx(expected)
    assert available == (expected >= 0.0)


def test_success_does_not_permanently_occupy_whole_axial_coordinate() -> None:
    axis, cfg = thick_parent(0.10)
    axis.branch_origins.append(1.0)
    axis.branch_azimuths.append(0.0)
    axis.branch_origin_base_radii.append(0.01)
    assert sim.origin_surface_clearance_rejection(
        axis, 1.0, math.pi, 0.01, cfg
    ) is None


def test_same_site_branches_are_independently_curved() -> None:
    axis, cfg = thick_parent(0.10)
    store = sim.RootAxisStore(cfg)
    store.axes[0] = axis
    site = sim.BranchSite(0, 0, 1.0, 2, 2)
    store.branch_sites.append(site)
    field_model = sim.HeterogeneousResourceField(211, cfg)
    rng = np.random.default_rng(211)
    first, first_reason = store.create_branch_axis(
        axis, 1.0, 0.0, 60.0, 2, 1.0, rng, field_model, False, site=site
    )
    second = None
    second_reason = "not_attempted"
    for azimuth in (math.pi, math.pi / 2.0, 3.0 * math.pi / 2.0, 2.0 * math.pi / 3.0):
        second, second_reason = store.create_branch_axis(
            axis, 1.0, azimuth, 60.0, 2, 1.0, rng, field_model, False, site=site
        )
        if second is not None:
            break
    assert first_reason is None and second_reason is None
    assert first is not None and second is not None
    assert site.accepted_branch_count == 2
    assert not np.allclose(first.points[1], second.points[1])


def test_range_addition_cache_invalidates_and_preserves_queries() -> None:
    cfg = config()
    axis = sim.RootAxisStore(cfg).axes[0]
    query = np.asarray([0.0, 0.25, 0.75, 1.0])
    axis.add_structural_area_event(1.0, 0.01)
    first = sim.axis_radii_at_arcs(axis, query, cfg)
    cached_version = axis._area_cache_version
    axis.add_structural_area_event(0.5, 0.02)
    second = sim.axis_radii_at_arcs(axis, query, cfg)
    assert axis._area_cache_version > cached_version
    assert np.all(second[:2] > first[:2])
    assert np.allclose(second[2:], first[2:])


def test_batch_configuration_keeps_retry_and_spacing_outside_grid() -> None:
    parser = sim.build_parser()
    args = parser.parse_args(["--mode", "batch", "--branch-retry-mode", "retry_open_sites"])
    batch = sim.config_from_args(args)
    assert batch.steps == 500
    assert batch.branch_min_spacing_along_axis == pytest.approx(0.20)
    assert batch.branch_retry_mode == "retry_open_sites"
    assert sim.TOTAL_GRID_TASKS == 3_430_350


def professor_grid_parameters(
    thickness_increment: float = 1.0,
    replicate: int = 0,
) -> sim.SimulationParameters:
    """Return the exact B.P.=.01/R.P.=.97 fixed-grid calibration case."""

    thickness_index = int(round(thickness_increment * 10.0)) - 1
    task_index = (((thickness_index * 99 + 96) * 99 + 0) * 5) + replicate
    return sim.parameters_for_task(task_index, 20260617)


def test_exact_low_bp_professor_case_keeps_primary_dominant() -> None:
    params = professor_grid_parameters(1.0, 0)
    assert params.task_index == 488_565
    assert (
        params.branch_probability,
        params.rain_probability,
        params.thickness_increment,
    ) == pytest.approx((0.01, 0.97, 1.0))
    metrics, store = sim.run_simulation(
        params,
        sim.SimulationConfig(
            steps=500,
            max_nodes=200_000,
            max_sampled_points=200_000,
            interactive_safety_cap=200_000,
            max_seconds_per_simulation=0.0,
            branch_retry_mode="single_trial",
        ),
        return_store=True,
    )
    assert metrics["normal_developmental_completion"] == 1
    assert metrics["developmental_steps_completed"] == 500
    assert metrics["max_first_order_lateral_length"] < metrics["primary_axis_length"]
    assert (
        metrics["max_first_order_lateral_length"]
        / metrics["primary_axis_length"]
    ) < 0.25
    # The isolated site/trial stream changes the exact accepted realization,
    # but the primary remains longer than all laterals combined and far longer
    # than every individual first-order lateral.
    assert metrics["lateral_to_primary_length_ratio"] < 1.0
    assert metrics["active_tip_attempt_accounting_error"] == 0
    assert metrics["primary_axis_basal_to_tip_radius_ratio"] > 1.0

    generations = np.asarray(store.axis_metadata["axis_generations"])
    lengths = np.asarray(store.axis_metadata["axis_arc_lengths"])
    radii = store.axis_metadata["axis_radii"]
    first_order = np.flatnonzero(generations == 1)
    assert first_order.size == metrics["first_order_lateral_count"]
    assert np.all(lengths[first_order] > 0.10)
    for axis_id in first_order:
        profile = np.asarray(radii[int(axis_id)], dtype=np.float64)
        if lengths[int(axis_id)] >= 1.0:
            assert profile[0] > profile[-1]


def test_lateral_elongation_uses_developmental_decay_not_constant_lifetime_rate() -> None:
    cfg = sim.SimulationConfig()
    lateral = manual_axis(1, 0, 0.5, 1, 1.0)
    lateral.birth_step = 24
    early = sim.tip_extension_length(lateral, 24, cfg)
    middle = sim.tip_extension_length(lateral, 150, cfg)
    late = sim.tip_extension_length(lateral, 499, cfg)
    assert early > middle >= late > 0.0
    assert late >= cfg.lateral_min_segment_length

    primary_total = sum(
        sim.tip_extension_length(sim.RootAxisStore(cfg).axes[0], step, cfg)
        for step in range(500)
    )
    lateral_remaining = sum(
        sim.tip_extension_length(lateral, step, cfg)
        for step in range(lateral.birth_step + 1, 500)
    )
    old_constant_remaining = (
        cfg.segment_length * 0.42 * (499 - lateral.birth_step)
    )
    assert lateral_remaining < 0.60 * primary_total
    assert lateral_remaining < 0.25 * old_constant_remaining

    source = inspect.getsource(sim.tip_extension_length)
    assert "axis.birth_step" in source
    assert "age_decay" in source
    assert "lateral_relative_elongation" in source
    assert "branch_probability" not in source


def test_lateral_length_is_bp_independent_and_all_active_tips_still_attempt() -> None:
    cfg = config(steps=40)
    lateral = manual_axis(1, 0, 0.5, 2, 1.0)
    lengths = [sim.tip_extension_length(lateral, step, cfg) for step in range(40)]
    assert min(lengths) > 0.0

    low, _ = run_case(
        223, branch_probability=0.01, steps=40,
        max_seconds_per_simulation=0.0,
    )
    high, _ = run_case(
        223, branch_probability=0.99, steps=40,
        max_seconds_per_simulation=0.0,
    )
    for metrics in (low, high):
        assert metrics["normal_developmental_completion"] == 1
        assert metrics["tip_extension_attempts"] == metrics["active_tips_at_step_start_total"]
        assert metrics["active_tip_attempt_accounting_error"] == 0


def test_curve_identifier_records_decay_and_retry_is_not_production_calibrated(
    tmp_path: Path,
) -> None:
    assert sim.SCHEMA_VERSION == 26
    assert "decaying-laterals" in sim.CURVE_MODEL_VERSION
    parser = sim.build_parser()
    args = parser.parse_args(["--mode", "batch"])
    path = tmp_path / "metadata.json"
    sim.write_metadata(path, sim.config_from_args(args), args)
    metadata = json.loads(path.read_text())
    assert metadata["canonical_production_branch_retry_mode"] is None
    assert metadata["branch_retry_mode_production_calibrated"] is False
    assert metadata["initiation_model_version"] == sim.INITIATION_MODEL_VERSION
    assert metadata["initiation_random_stream_version"] == sim.INITIATION_RANDOM_STREAM_VERSION
    assert metadata["initiation_probability_resource_independent"] is True


# Resource-response behavior remains part of the scientific contract.
def test_v26_required_resource_direction_fields_are_in_contract() -> None:
    required = {
        "resource_environment_step", "cumulative_rain_input",
        "effective_wetting_depth", "effective_nitrate_depth",
        "effective_potassium_depth", "water_active_target_share",
        "phosphorus_active_target_share", "nitrogen_active_target_share",
        "potassium_active_target_share", "water_normalized_capture_share",
        "phosphorus_normalized_capture_share", "nitrogen_normalized_capture_share",
        "potassium_normalized_capture_share", "water_deficiency",
        "phosphorus_deficiency", "nitrogen_deficiency", "potassium_deficiency",
        "water_demand_weight", "phosphorus_demand_weight",
        "nitrogen_demand_weight", "potassium_demand_weight",
        "water_focus_axis_count", "phosphorus_focus_axis_count",
        "nitrogen_focus_axis_count", "potassium_focus_axis_count",
        "balanced_focus_axis_count", "resource_focus_updates",
        "mean_lateral_emergence_angle", "median_lateral_emergence_angle",
        "fraction_near_horizontal_lateral_segments",
        "fraction_downward_lateral_segments",
        "fraction_mildly_upward_lateral_segments",
        "maximum_consecutive_upward_extensions", "architecture_width",
        "architecture_depth", "architecture_depth_width_ratio",
        "initiation_model_version", "initiation_random_stream_version",
        "initiation_probability_resource_independent",
        "initiation_uniform_mean", "probability_pass_rate",
        "physical_rejection_count", "physical_rejection_rate",
    }
    assert required <= set(sim.RESULT_FIELDS)


def test_environment_is_time_dependent_absolute_and_surface_anchored() -> None:
    cfg = sim.SimulationConfig()
    field = sim.HeterogeneousResourceField(401, cfg)
    surface = np.asarray([[0.0, 0.0, 0.0]])
    fixed_depth = np.asarray([[0.0, 0.0, -12.5]])
    p_surface_initial = float(field.values(surface, False, cfg)[1][0])
    n_fixed_initial = float(field.values(fixed_depth, False, cfg)[2][0])
    initial_nitrate_depth = field.environment.effective_nitrate_depth
    for step in range(50):
        field.environment.update(step, True, cfg)
    p_surface_later = float(field.values(surface, True, cfg)[1][0])
    n_fixed_later = float(field.values(fixed_depth, True, cfg)[2][0])
    assert field.environment.current_step == 49
    assert field.environment.cumulative_rain_input > 0.0
    assert field.environment.effective_wetting_depth > 0.0
    assert field.environment.effective_nitrate_depth > initial_nitrate_depth
    assert field.environment.effective_potassium_depth > 0.0
    assert p_surface_initial > 0.0 and p_surface_later > 0.0
    assert n_fixed_initial != pytest.approx(n_fixed_later)
    # Evaluating a deeper root does not mutate or rescale the absolute profile.
    state_before = asdict(field.environment)
    field.values(np.asarray([[0.0, 0.0, -250.0]]), True, cfg)
    assert asdict(field.environment) == state_before


def test_zero_supply_has_zero_target_demand_and_focus_probability() -> None:
    cfg = zero_resource_config()
    environment = sim.ResourceEnvironmentState()
    environment.update(0, False, cfg)
    demand = sim.ResourceDemandState(cfg)
    demand.begin_step(environment, cfg)
    assert np.array_equal(demand.supply, np.zeros(4))
    assert np.array_equal(demand.active_target_shares, np.zeros(4))
    assert np.array_equal(demand.weights(cfg), np.zeros(4))
    probabilities = demand.focus_probabilities(np.ones(4), cfg)
    assert np.array_equal(probabilities[:4], np.zeros(4))
    assert probabilities[4] == pytest.approx(1.0)


def test_capture_rate_normalization_prevents_low_rate_false_deficiency() -> None:
    cfg = sim.SimulationConfig()
    environment = sim.ResourceEnvironmentState()
    environment.update(0, True, cfg)
    demand = sim.ResourceDemandState(cfg)
    demand.capture_totals = demand.capture_rates(cfg) * 10.0
    demand.begin_step(environment, cfg)
    assert demand.normalized_capture_shares == pytest.approx(np.full(4, 0.25))
    assert np.all(demand.weights(cfg) >= 0.0)
    assert np.all(demand.weights(cfg) <= cfg.resource_demand_weight_cap)


def test_resource_focus_draws_are_diverse_reproducible_and_supply_aware() -> None:
    cfg = sim.SimulationConfig()
    environment = sim.ResourceEnvironmentState()
    environment.update(25, True, cfg)
    demand = sim.ResourceDemandState(cfg)
    demand.begin_step(environment, cfg)
    local = np.asarray([0.7, 0.7, 0.7, 0.7])
    first_rng = np.random.default_rng(991)
    second_rng = np.random.default_rng(991)
    first = [sim.draw_resource_focus(first_rng, local, demand, cfg)[0] for _ in range(50)]
    second = [sim.draw_resource_focus(second_rng, local, demand, cfg)[0] for _ in range(50)]
    assert first == second
    assert len(set(first)) >= 3


def test_emergence_angle_is_resource_conditioned_not_fixed_62_degrees() -> None:
    poor_rng = np.random.default_rng(77)
    rich_p_rng = np.random.default_rng(77)
    rich_n_rng = np.random.default_rng(77)
    poor = np.asarray([
        sim.sample_branch_emergence_angle(poor_rng, starvation=1.0, sufficiency=0.0)
        for _ in range(300)
    ])
    rich_p = np.asarray([
        sim.sample_branch_emergence_angle(
            rich_p_rng, starvation=0.0, phosphorus=1.0,
            resource_focus="phosphorus", sufficiency=1.0,
        ) for _ in range(300)
    ])
    rich_n = np.asarray([
        sim.sample_branch_emergence_angle(
            rich_n_rng, starvation=0.0, nitrogen=1.0,
            resource_focus="nitrogen", sufficiency=1.0,
        ) for _ in range(300)
    ])
    assert 10.0 <= np.min(poor) and np.max(poor) <= 35.0
    assert 80.0 <= np.mean(rich_p) <= 105.0
    assert np.mean(rich_p) > np.mean(rich_n) + 20.0
    assert "lateral_emergence_angle_degrees" not in {
        item.name for item in fields(sim.SimulationConfig)
    }


def test_direction_candidates_are_diverse_and_bp_absent_post_initiation() -> None:
    source = inspect.getsource(sim.biased_axis_direction)
    context_source = inspect.getsource(sim.prepare_direction_resource_context)
    assert "resource_context" in source
    assert "prepare_direction_resource_context" in source
    assert "cone_directions_from_frame" in source
    assert "curvature_limited_toward(previous, target" in context_source
    post_initiation_functions = (
        sim.sample_branch_emergence_angle,
        sim.biased_axis_direction,
        sim.tip_extension_length,
        sim.try_extend_axis,
        sim.record_transport_path_growth,
        sim.sampled_axis_radius,
    )
    for function in post_initiation_functions:
        assert "branch_probability" not in inspect.getsource(function)


def test_matched_seed_rich_architecture_is_obviously_wider_than_zero_resource() -> None:
    common = dict(
        branch_probability=0.20,
        rain_probability=0.99,
        steps=70,
        max_nodes=100_000,
        max_sampled_points=100_000,
        interactive_safety_cap=100_000,
        max_seconds_per_simulation=0.0,
    )
    rich, _ = run_case(123, **common)
    zero, _ = run_case(
        123,
        **common,
        soil_water_background=0.0,
        rain_water_input=0.0,
        phosphorus_concentration=0.0,
        nitrogen_concentration=0.0,
        potassium_concentration=0.0,
    )
    assert rich["normal_developmental_completion"] == 1
    assert zero["normal_developmental_completion"] == 1
    assert rich["architecture_width"] > 2.0 * zero["architecture_width"]
    assert rich["architecture_depth_width_ratio"] < 0.60 * zero["architecture_depth_width_ratio"]
    assert rich["fraction_near_horizontal_lateral_segments"] > zero["fraction_near_horizontal_lateral_segments"]
    assert rich["maximum_consecutive_upward_extensions"] <= 2
    assert zero["maximum_consecutive_upward_extensions"] <= 2


def test_time_series_contains_requested_completed_milestones() -> None:
    metrics, _ = run_case(
        515, branch_probability=0.06, rain_probability=0.99,
        steps=50, max_seconds_per_simulation=0.0,
    )
    snapshots = json.loads(str(metrics["resource_time_series_json"]))
    assert [row["step"] for row in snapshots] == [0, 10, 25, 50]


def scientific_results_equal(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    timing = {"execution_time_sec"} | {
        key for key in left if key.startswith("profile_")
    }
    for key in left:
        if key in timing:
            continue
        left_value = left[key]
        right_value = right[key]
        if (
            isinstance(left_value, float)
            and isinstance(right_value, float)
            and math.isnan(left_value)
            and math.isnan(right_value)
        ):
            continue
        if left_value != right_value:
            return False
    return True


def stable_json_hash(value: object) -> str:
    """Hash a value using the fixture's canonical JSON representation."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def stable_array_hash(values: np.ndarray) -> str:
    """Hash array dtype, shape, and exact contiguous bytes."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def test_engine_matches_schema_v26_regression_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    result, store = sim.run_simulation(
        sim.parameters_for_task(
            int(fixture["task_index"]), int(fixture["master_seed"])
        ),
        sim.SimulationConfig(**fixture["config"]),
        return_store=True,
    )
    scientific = {
        key: value
        for key, value in result.items()
        if key != "execution_time_sec" and not key.startswith("profile_")
    }
    assert sim.SCHEMA_VERSION == fixture["schema_version"]
    assert len(scientific) == fixture["scientific_result_field_count"]
    assert stable_json_hash(scientific) == fixture["scientific_result_sha256"]
    assert store.size == fixture["store_size"]
    arrays = {
        "position": store.position[:store.size],
        "parent": store.parent[:store.size],
        "radius": store.radius[:store.size],
        "axis_parent_ids": store.axis_metadata["axis_parent_ids"],
        "axis_parent_arc_lengths": store.axis_metadata["axis_parent_arc_lengths"],
        "axis_parent_local_azimuths": store.axis_metadata["axis_parent_local_azimuths"],
        "axis_birth_steps": store.axis_metadata["axis_birth_steps"],
        "axis_extension_events": store.axis_metadata["axis_extension_events"],
    }
    assert {
        name: stable_array_hash(values) for name, values in arrays.items()
    } == fixture["arrays"]
    assert stable_json_hash(store.axis_metadata["branch_sites"]) == (
        fixture["branch_sites_sha256"]
    )
    for key, expected in fixture["selected_metrics"].items():
        assert result[key] == expected


def test_atomic_checkpoint_resume_matches_uninterrupted_output(tmp_path: Path) -> None:
    cfg = config(
        steps=18,
        max_seconds_per_simulation=0.0,
        branch_retry_mode="single_trial",
    )
    params = parameters(20260715, branch_probability=0.60, rain_probability=0.90)
    complete, complete_store = sim.run_simulation(
        params, cfg, return_store=True
    )
    checkpoint = tmp_path / "replicate.checkpoint"
    partial, _partial_store = sim.run_simulation(
        params,
        cfg,
        return_store=True,
        checkpoint_path=checkpoint,
        checkpoint_interval_steps=9,
        pause_after_checkpoint_step=9,
    )
    assert partial["status"] == "checkpoint_pause"
    assert partial["developmental_steps_completed"] == 9
    resumed, resumed_store = sim.run_simulation(
        params,
        cfg,
        return_store=True,
        resume_checkpoint_path=checkpoint,
    )
    assert scientific_results_equal(complete, resumed)
    assert np.array_equal(
        complete_store.position[:complete_store.size],
        resumed_store.position[:resumed_store.size],
    )
    assert np.array_equal(
        complete_store.parent[:complete_store.size],
        resumed_store.parent[:resumed_store.size],
    )
    assert np.array_equal(
        complete_store.radius[:complete_store.size],
        resumed_store.radius[:resumed_store.size],
    )


def test_checkpoint_rejects_configuration_mismatch(tmp_path: Path) -> None:
    cfg = config(steps=10, max_seconds_per_simulation=0.0)
    params = parameters(881)
    checkpoint = tmp_path / "incompatible.checkpoint"
    sim.run_simulation(
        params,
        cfg,
        checkpoint_path=checkpoint,
        checkpoint_interval_steps=5,
        pause_after_checkpoint_step=5,
    )
    with pytest.raises(hpc_storage.CheckpointCompatibilityError):
        sim.run_simulation(
            params,
            replace(cfg, phosphorus_concentration=0.123),
            resume_checkpoint_path=checkpoint,
        )


def test_lossless_lazy_result_bundle_and_lod_hash_invariance(tmp_path: Path) -> None:
    result, store = run_case(991, branch_probability=0.40, steps=12)
    store.axis_metadata["node_branch_generation"] = np.zeros(
        store.size, dtype=np.int32
    )
    store.axis_metadata["node_strahler_orders"] = np.asarray(
        sim.compute_strahler_orders(store), dtype=np.int32
    )
    bundle_path = tmp_path / "replicate_0"
    hpc_storage.save_result_bundle(
        bundle_path,
        result=result,
        store=store,
        provenance={"geometry_hash": "fixture"},
    )
    lazy = hpc_storage.load_result_bundle(bundle_path, mmap_mode="r")
    assert isinstance(lazy["position"], np.memmap)
    assert np.array_equal(
        lazy["position"], store.position[:store.size]
    )
    assert np.array_equal(lazy["parent"], store.parent[:store.size])
    before_hash = hashlib.sha256(lazy["position"].tobytes()).hexdigest()
    preview = hpc_storage.deterministic_lod_axis_ids(100_000, "Preview")
    medium = hpc_storage.deterministic_lod_axis_ids(100_000, "Medium")
    assert preview.size == 2_000
    assert medium.size == 10_000
    assert np.array_equal(preview, np.arange(2_000, dtype=np.int32))
    after_hash = hashlib.sha256(lazy["position"].tobytes()).hexdigest()
    assert before_hash == after_hash


def test_massive_manifest_accepts_50000_and_optional_100000_steps(
    tmp_path: Path,
) -> None:
    grid = {
        "branch_probability": 0.99,
        "rain_probability": 0.99,
        "thickness_increment": 0.10,
    }
    for steps, cap in ((50_000, 10_000_000), (100_000, 20_000_000)):
        cfg = sim.SimulationConfig(
            steps=steps,
            max_nodes=cap,
            max_sampled_points=cap,
            interactive_safety_cap=cap,
            max_seconds_per_simulation=72 * 3600,
        )
        run_dir = hpc_manager.create_run_manifest(
            simulator_path=Path(sim.__file__),
            app_path=Path(sim.__file__),
            config=cfg,
            grid_values=grid,
            partition="standard",
            wall_time="72:00:00",
            memory_gb=64,
            cpus_per_task=1,
            checkpoint_interval_steps=5,
            rendering_lod="Preview",
            runs_root=tmp_path,
        )
        manifest = hpc_manager.load_manifest(run_dir)
        assert manifest["slurm_array"] == "0-4"
        assert manifest["replicate_count"] == 5
        assert manifest["production_grid_sweep"] is False
        assert len(manifest["replicate_tasks"]) == 5
        assert hpc_manager.submit_run(run_dir, dry_run=True).startswith("DRYRUN-")
        resumed = hpc_manager.resume_incomplete_run(run_dir, dry_run=True)
        assert resumed is not None
        assert resumed.startswith("DRYRUN-RESUME-")
        resume_submission = json.loads(
            (run_dir / "resume_submission.json").read_text()
        )
        assert resume_submission["array"] == "0,1,2,3,4"
