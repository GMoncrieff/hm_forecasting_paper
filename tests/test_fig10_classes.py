"""Boundary tests for the 30% protection-target classifier.

The classifier is the only place in fig10 where a judgement is made rather than
a sum, and every one of its five outcomes turns on an inequality. Getting the
order of the tests wrong swaps 'infeasible' for 'at risk' on exactly the
ecoregions where the distinction matters most, and nothing downstream would
notice.
"""
import pytest

import config
from figures import protection_target_class as classify

TARGET = config.PROTECTION_TARGET


def test_already_met_takes_priority():
    assert classify(TARGET, 0, 0, 0) == 'already met'
    assert classify(TARGET + 20, 0, 0, 0) == 'already met'
    # even with no natural land left at all
    assert classify(TARGET + 1, 0.0, 0.0, 0.0) == 'already met'


def test_exactly_at_the_target_counts_as_met():
    """>= not >, so an ecoregion sitting precisely on 30% is not 'at risk'."""
    assert classify(TARGET, 0, 0, 0) == 'already met'
    assert classify(TARGET - 1e-9, 100, 0, 0) == 'feasible'


def test_infeasible_when_natural_land_runs_out():
    """Protected plus every scrap of natural land still short of the target."""
    assert classify(5.0, 3.0, 3.0, 3.0) == 'infeasible on natural land'
    # 5 + 24.9 = 29.9 < 30
    assert classify(5.0, 10.0, 10.0, 4.9) == 'infeasible on natural land'


def test_infeasible_boundary_is_exact():
    """Reaching the target exactly is feasible, not infeasible."""
    assert classify(5.0, 10.0, 10.0, 5.0) != 'infeasible on natural land'
    assert classify(5.0, 10.0, 10.0, 4.999) == 'infeasible on natural land'


def test_feasible_uses_only_the_safest_land():
    assert classify(10.0, 20.0, 0.0, 0.0) == 'feasible'
    assert classify(0.0, 30.0, 50.0, 10.0) == 'feasible'


def test_tight_needs_the_middle_class():
    """Safe land alone falls short; safe plus middling gets there."""
    assert classify(10.0, 19.9, 5.0, 0.0) == 'tight'
    assert classify(0.0, 29.0, 1.0, 0.0) == 'tight'


def test_at_risk_needs_the_worst_class():
    assert classify(10.0, 5.0, 5.0, 30.0) == 'at risk'
    assert classify(0.0, 10.0, 10.0, 15.0) == 'at risk'


def test_the_four_reachable_classes_are_ordered_by_how_much_risk_they_need():
    """Moving natural land from safe to risky can only worsen the class."""
    order = ['feasible', 'tight', 'at risk']
    seen = [classify(10.0, lo, mid, hi) for lo, mid, hi in
            ((25.0, 5.0, 5.0), (15.0, 15.0, 5.0), (5.0, 5.0, 25.0))]
    assert seen == order


@pytest.mark.parametrize('target', [10.0, 30.0, 50.0])
def test_target_is_configurable(target):
    assert classify(target, 0, 0, 0, target=target) == 'already met'
    assert classify(target - 0.001, target, 0, 0, target=target) == 'feasible'


def test_every_class_is_declared_in_config():
    """config.TARGET_CLASSES drives the map legend and colours."""
    produced = {
        classify(35, 0, 0, 0), classify(1, 2, 3, 4), classify(10, 25, 0, 0),
        classify(10, 5, 20, 0), classify(10, 5, 5, 20),
    }
    assert produced == set(config.TARGET_CLASSES)
    assert set(config.TARGET_CLASS_COLORS) == set(config.TARGET_CLASSES)
    assert set(config.TARGET_CLASS_DESCRIPTIONS) == set(config.TARGET_CLASSES)
