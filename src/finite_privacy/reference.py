"""Bounded independent reference calculations for small finite experiments."""

from fractions import Fraction
from itertools import product


def exhaustive_alpha(p, q, target_delta):
    """Solve every event inequality separately, then take their largest bound."""
    if len(p) != len(q) or not 1 <= len(p) <= 12:
        raise ValueError("event enumeration needs equal supports of size 1..12")
    required = Fraction(1)
    for mask in range(1 << len(p)):
        p_event = sum((v for i, v in enumerate(p) if mask & (1 << i)), Fraction())
        q_event = sum((v for i, v in enumerate(q) if mask & (1 << i)), Fraction())
        for numerator, denominator in ((p_event, q_event), (q_event, p_event)):
            if denominator == 0:
                if numerator > target_delta:
                    return None
            else:
                required = max(required, (numerator - target_delta) / denominator)
    return required


def exhaustive_events(p, q, alpha):
    """Enumerate EVERY event subset, rather than thresholding each mass pair."""
    if len(p) != len(q) or not 1 <= len(p) <= 12:
        raise ValueError("event enumeration needs equal supports of size 1..12")
    best_forward = best_reverse = Fraction()
    for mask in range(1 << len(p)):
        p_event = sum((v for i, v in enumerate(p) if mask & (1 << i)), Fraction())
        q_event = sum((v for i, v in enumerate(q) if mask & (1 << i)), Fraction())
        best_forward = max(best_forward, p_event - alpha * q_event)
        best_reverse = max(best_reverse, q_event - alpha * p_event)
    return best_forward, best_reverse


def enumerate_transcripts(channels, *, limit=65536):
    """Explicit output tuples and their probabilities; no ratio compression."""
    channels = tuple(channels)
    count = 1
    for channel in channels:
        count *= len(channel.p)
        if count > limit:
            raise ValueError("transcript reference exceeds its bounded size")
    p, q = [], []
    for outputs in product(*(range(len(channel.p)) for channel in channels)):
        p_mass = q_mass = Fraction(1)
        for channel, index in zip(channels, outputs):
            p_mass *= channel.p[index]
            q_mass *= channel.q[index]
        p.append(p_mass)
        q.append(q_mass)
    return tuple(p), tuple(q)
