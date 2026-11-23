import pytest

from proxy_function_gap.constraints import UTRConstraints, utr_constraint_mask


def test_combined_utr_hard_constraints():
    constraints = UTRConstraints(
        min_length=6,
        max_length=6,
        gc_min=0.2,
        gc_max=0.6,
        forbid_uaug=True,
        max_poly_u=3,
    )
    sequences = ("AUGCCC", "UUUUCC", "ACGUAC", "AC")
    assert utr_constraint_mask(sequences, constraints) == (False, False, True, False)


def test_gc_and_length_boundaries_are_inclusive():
    constraints = UTRConstraints(min_length=2, max_length=2, gc_min=1.0, gc_max=1.0)
    assert utr_constraint_mask(("GG",), constraints) == (True,)


def test_zero_length_poly_u_limit_rejects_any_u():
    assert utr_constraint_mask(("ACG", "ACU"), UTRConstraints(max_poly_u=0)) == (True, False)


def test_invalid_alphabet_is_not_silently_masked():
    with pytest.raises(ValueError, match="uppercase"):
        utr_constraint_mask(("ACGT",))


def test_invalid_constraint_configuration_is_rejected():
    with pytest.raises(ValueError):
        UTRConstraints(min_length=5, max_length=4)
    with pytest.raises(ValueError):
        UTRConstraints(gc_min=0.8, gc_max=0.2)
    with pytest.raises(ValueError):
        UTRConstraints(max_poly_u=-1)
