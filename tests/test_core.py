from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from math import comb
import random

import pytest

from finite_privacy import Channel, compose_independent, randomized_response
from finite_privacy.demo import shared_coin_channels
from finite_privacy.reference import enumerate_transcripts, exhaustive_alpha, exhaustive_events


@pytest.mark.parametrize("seed", range(10))
def test_profiles_and_calibration_against_every_event(seed):
    rng = random.Random(seed)
    a, b = [rng.randrange(5) for _ in range(5)], [rng.randrange(5) for _ in range(5)]
    a[0] += 1
    b[0] += 1
    p, q = tuple(F(x, sum(a)) for x in a), tuple(F(x, sum(b)) for x in b)
    channel = Channel(p, q)
    for alpha in (F(1), F(3, 2), F(3), F(10)):
        result = channel.profile(alpha)
        assert (result.forward_delta, result.reverse_delta) == exhaustive_events(p, q, alpha)
        assert sum((p[i] - alpha * q[i] for i in result.forward_event), F()) == result.forward_delta
        assert sum((q[i] - alpha * p[i] for i in result.reverse_event), F()) == result.reverse_delta
    for delta in (F(0), F(1, 20), F(1, 4), F(1)):
        calibrated = channel.calibrate(delta)
        assert calibrated == exhaustive_alpha(p, q, delta)
        if calibrated is not None:
            assert channel.profile(calibrated).delta <= delta
            if calibrated > 1:
                assert channel.profile((calibrated + 1) / 2).delta > delta


def test_randomized_response_closed_form_and_exact_boundary():
    rr = randomized_response(F(3, 4))
    assert rr.profile(1).delta == F(1, 2)
    assert rr.profile(2).delta == F(1, 4)
    assert rr.profile(3).delta == 0
    assert rr.profile(3).forward_event == ()
    assert rr.calibrate(F(1, 100)) == F(74, 25)
    assert rr.calibrate(0) == 3
    assert rr.profile(3 - F(1, 10**100)).delta == F(1, 4 * 10**100)


def test_asymmetric_directions_must_both_be_checked():
    channel = Channel((F(1, 2), F(1, 2)), (F(1, 4), F(3, 4)))
    p = channel.profile(F(3, 2))
    assert p.forward_delta == F(1, 8)
    assert p.reverse_delta == 0
    assert channel.calibrate(0) == 2
    assert Channel(channel.q, channel.p).profile(F(3, 2)).delta == p.delta


def test_total_variation_and_profile_shape():
    channel = Channel((F(1, 10), F(7, 10), F(2, 10)), (F(5, 10), F(1, 10), F(4, 10)))
    assert channel.profile(1).delta == sum(abs(p - q) for p, q in zip(channel.p, channel.q)) / 2
    vals = [channel.profile(F(i, 2)).delta for i in range(2, 20)]
    assert all(x >= y for x, y in zip(vals, vals[1:]))
    assert all(x + z >= 2 * y for x, y, z in zip(vals, vals[1:], vals[2:]))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8])
def test_composition_against_transcripts_and_binomial_formula(n):
    rr = randomized_response(F(3, 4))
    composed = compose_independent(rr for _ in range(n))
    raw = Channel(*enumerate_transcripts([rr] * n))
    assert len(composed.channel.p) == n + 1
    assert composed.pair_operations == n * (n + 1)
    assert composed.channel.calibrate(0) == 3**n
    for alpha in (F(1), F(3), F(9), F(3**n)):
        exact = sum((comb(n, k) * max(F(3**k, 4**n) - alpha * F(3**(n-k), 4**n), 0)
                     for k in range(n + 1)), F())
        assert composed.channel.profile(alpha).delta == exact == raw.profile(alpha).delta


def test_heterogeneous_composition_and_permutation():
    channels = [randomized_response(F(2, 3)), randomized_response(F(4, 5)),
                Channel((F(1, 2), F(1, 3), F(1, 6)), (F(1, 4), F(1, 4), F(1, 2)))]
    composed = compose_independent(channels).channel
    raw = Channel(*enumerate_transcripts(channels))
    assert composed == compose_independent(reversed(channels)).channel
    for alpha in (F(1), F(5, 2), F(7), F(32)):
        assert composed.profile(alpha).delta == raw.profile(alpha).delta
    for delta in (F(0), F(1, 100), F(1, 5)):
        assert composed.calibrate(delta) == raw.calibrate(delta)


def test_compression_removes_only_impossible_or_equal_ratio_states():
    channel = Channel((F(1, 2), F(1, 4), 0, F(1, 4), 0), (F(1, 4), F(1, 8), F(5, 8), 0, 0))
    compressed = channel.compress()
    assert compressed == Channel((0, F(3, 4), F(1, 4)), (F(5, 8), F(3, 8), 0))
    for alpha in (1, 2, 10**50):
        assert channel.profile(alpha).delta == compressed.profile(alpha).delta
    assert compressed.compress() == compressed
    assert channel.p[-1] == 0  # input not mutated
    with pytest.raises(FrozenInstanceError):
        channel.p = (1,)


def test_singular_mass_and_zero_times_infinity_composition():
    channel = Channel((F(1, 4), F(3, 4), 0), (0, F(1, 2), F(1, 2)))
    assert channel.delta_floor == F(1, 2)
    assert channel.calibrate(F(49, 100)) is None
    assert channel.calibrate(F(1, 2)) == 1
    reveal = Channel((1, 0), (0, 1))
    result = compose_independent((reveal, reveal)).channel
    assert len(result.p) == 2  # crossed transcript outcomes have mass zero under both inputs
    assert result.delta_floor == 1
    assert result.calibrate(0) is None
    assert result.calibrate(1) == 1


def test_calibration_at_singular_floor_above_one():
    channel = Channel((F(1, 10), F(8, 10), F(1, 10)), (0, F(2, 10), F(8, 10)))
    assert channel.delta_floor == F(1, 10)
    alpha = channel.calibrate(F(1, 10))
    assert alpha == 7
    assert channel.profile(alpha).delta == F(1, 10)
    assert channel.profile(alpha - F(1, 100)).delta > F(1, 10)


def test_postprocessing_identity_permutation_constant_and_random_kernel():
    c = Channel((F(1, 4), F(3, 4)), (F(4, 5), F(1, 5)))
    assert c.postprocess(((1, 0), (0, 1))) == c
    flipped = c.postprocess(((0, 1), (1, 0)))
    constant = c.postprocess(((1,), (1,)))
    mixed = c.postprocess(((F(1, 2), F(1, 3), F(1, 6)), (F(1, 4), F(1, 4), F(1, 2))))
    for alpha in (F(1), F(3, 2), F(3), F(10)):
        assert flipped.profile(alpha).delta == c.profile(alpha).delta
        assert constant.profile(alpha).delta == 0
        assert mixed.profile(alpha).delta <= c.profile(alpha).delta


def test_shared_randomness_counterexample():
    joint, first, second = shared_coin_channels()
    assert first == second == Channel((F(1, 2), F(1, 2)), (F(1, 2), F(1, 2)))
    assert compose_independent((first, second)).channel.profile(1).delta == 0
    assert joint.profile(10**100).delta == 1
    assert joint.profile(1).forward_event == (0, 3)
    parity = joint.postprocess(((1, 0), (0, 1), (0, 1), (1, 0)))
    assert parity == Channel((1, 0), (0, 1))


def test_resource_limits_fail_without_approximation():
    rr = randomized_response()
    with pytest.raises(ValueError, match="max_states"):
        compose_independent([rr] * 4, max_states=3)
    with pytest.raises(ValueError, match="max_pair_operations"):
        compose_independent([rr] * 4, max_pair_operations=10)
    with pytest.raises(ValueError):
        compose_independent([rr], max_states=True)
    with pytest.raises(ValueError):
        compose_independent([None])
    with pytest.raises(ValueError):
        enumerate_transcripts([rr] * 17)
    with pytest.raises(ValueError):
        exhaustive_events([F(1, 13)] * 13, [F(1, 13)] * 13, F(1))


@pytest.mark.parametrize("p,q", [((), ()), ((1,), (1, 0)), ((F(1, 2),), (1,)),
    ((-1, 2), (1, 0)), ((True,), (1,)), ((1.0,), (1,)), (("1",), (1,)),
    ((float("nan"),), (1,)), ((float("inf"),), (1,))])
def test_invalid_channels(p, q):
    with pytest.raises(ValueError):
        Channel(p, q)


@pytest.mark.parametrize("alpha", [0, -1, F(1, 2), 1.5, True, "2"])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        randomized_response().profile(alpha)


@pytest.mark.parametrize("delta", [-1, F(11, 10), 0.1, True, "0"])
def test_invalid_target(delta):
    with pytest.raises(ValueError):
        randomized_response().calibrate(delta)


@pytest.mark.parametrize("kernel", [(), ((1,),), ((1, 0), (1,)), ((0, 0), (1, 0)),
    ((F(1, 2), F(1, 2)), (True, 0)), ((-1, 2), (1, 0)), ((), ())])
def test_invalid_kernel(kernel):
    with pytest.raises(ValueError):
        randomized_response().postprocess(kernel)
