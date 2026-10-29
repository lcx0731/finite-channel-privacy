"""Finite binary-input privacy channels with exact rational accounting.

P and Q are output laws for ONE specified neighboring input pair. Composition
assumes conditional independence given that input. This module cannot verify
those modeling assumptions or turn sampled frequencies into DP guarantees.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction


def _fraction(value, name, *, lower=0, upper=None):
    if type(value) not in (int, Fraction):
        raise ValueError(f"{name} must be a built-in int or Fraction, not a float or bool")
    value = Fraction(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def exact_json(value):
    """Convert exact rational values to strings without decimal rounding."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {k: exact_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_json(v) for v in value]
    return value


@dataclass(frozen=True)
class Profile:
    alpha: Fraction
    forward_delta: Fraction
    reverse_delta: Fraction
    forward_event: tuple[int, ...]
    reverse_event: tuple[int, ...]

    @property
    def delta(self):
        return max(self.forward_delta, self.reverse_delta)

    def to_dict(self):
        return exact_json({**asdict(self), "delta": self.delta})


@dataclass(frozen=True)
class Channel:
    """Output masses P(.|input=0), Q(.|input=1); indices identify outputs."""

    p: tuple[Fraction, ...]
    q: tuple[Fraction, ...]

    def __post_init__(self):
        p, q = tuple(self.p), tuple(self.q)
        if not p or len(p) != len(q):
            raise ValueError("p and q must be nonempty and have equal length")
        p = tuple(_fraction(x, "p probability", upper=1) for x in p)
        q = tuple(_fraction(x, "q probability", upper=1) for x in q)
        if sum(p) != 1 or sum(q) != 1:
            raise ValueError("each probability row must sum exactly to one")
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "q", q)

    def profile(self, alpha=1):
        """Tight two-direction delta at alpha=exp(epsilon), with event witnesses."""
        alpha = _fraction(alpha, "alpha", lower=1)
        forward = tuple(i for i, (p, q) in enumerate(zip(self.p, self.q)) if p > alpha * q)
        reverse = tuple(i for i, (p, q) in enumerate(zip(self.p, self.q)) if q > alpha * p)
        return Profile(alpha, sum((self.p[i] - alpha * self.q[i] for i in forward), Fraction()),
                       sum((self.q[i] - alpha * self.p[i] for i in reverse), Fraction()), forward, reverse)

    @property
    def delta_floor(self):
        """Smallest achievable delta at any finite alpha, due to singular mass."""
        return max(sum((p for p, q in zip(self.p, self.q) if q == 0), Fraction()),
                   sum((q for p, q in zip(self.p, self.q) if p == 0), Fraction()))

    def calibrate(self, target_delta):
        """Smallest exact alpha>=1 giving delta<=target; None if impossible.

        Each directional profile is piecewise linear in alpha. Sweep its
        rational breakpoints and solve the active linear segment exactly.
        """
        target = _fraction(target_delta, "target_delta", upper=1)

        def directional(p_values, q_values):
            singular = sum((p for p, q in zip(p_values, q_values) if q == 0), Fraction())
            if target < singular:
                return None
            active = sorted((p / q, p, q) for p, q in zip(p_values, q_values) if p > q > 0)
            intercept = singular + sum((p for _, p, _ in active), Fraction())
            slope = sum((q for _, _, q in active), Fraction())
            if intercept - slope <= target:
                return Fraction(1)
            for ratio, p, q in active:
                candidate = (intercept - target) / slope
                if candidate <= ratio:
                    return candidate
                intercept -= p
                slope -= q
            raise ArithmeticError("unreachable calibration state")

        forward, reverse = directional(self.p, self.q), directional(self.q, self.p)
        return None if forward is None or reverse is None else max(forward, reverse)

    def postprocess(self, kernel):
        """Apply the SAME finite stochastic output kernel under both inputs."""
        rows = tuple(tuple(row) for row in kernel)
        if len(rows) != len(self.p) or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("kernel must have one equal-width nonempty row per channel output")
        rows = tuple(tuple(_fraction(x, "kernel probability", upper=1) for x in row) for row in rows)
        if any(sum(row) != 1 for row in rows):
            raise ValueError("each kernel row must sum exactly to one")
        p = tuple(sum((self.p[i] * rows[i][j] for i in range(len(rows))), Fraction()) for j in range(len(rows[0])))
        q = tuple(sum((self.q[i] * rows[i][j] for i in range(len(rows))), Fraction()) for j in range(len(rows[0])))
        return Channel(p, q)

    def compress(self, *, max_states=100_000):
        """Group equal likelihood ratios; preserves every privacy profile.

        New indices refer to ratio buckets, not original output symbols.
        States with p=q=0 disappear. Singular p>0,q=0 states form one bucket.
        """
        return _group(zip(self.p, self.q), _positive_int(max_states, "max_states"))

    def to_dict(self):
        return exact_json({"p": self.p, "q": self.q})


def _group(pairs, max_states):
    groups = {}
    for p, q in pairs:
        if p == q == 0:
            continue
        ratio = p / q if q else None
        old_p, old_q = groups.get(ratio, (Fraction(), Fraction()))
        groups[ratio] = (old_p + p, old_q + q)
        if len(groups) > max_states:
            raise ValueError("composition exceeds max_states; no approximation was substituted")
    ordered = sorted(groups, key=lambda r: (r is None, r if r is not None else 0))
    return Channel(tuple(groups[r][0] for r in ordered), tuple(groups[r][1] for r in ordered))


@dataclass(frozen=True)
class Composition:
    channel: Channel
    releases: int
    states_per_step: tuple[int, ...]
    pair_operations: int


def compose_independent(channels, *, max_states=100_000, max_pair_operations=1_000_000):
    """Product conditional laws for a shared binary input, compressed exactly.

    Conditional independence is an assumption, not inferred from marginals.
    Empty composition returns a constant channel. Work and state caps raise
    errors instead of silently truncating or merging unequal ratios.
    """
    max_states = _positive_int(max_states, "max_states")
    max_pair_operations = _positive_int(max_pair_operations, "max_pair_operations")
    current = Channel((1,), (1,))
    sizes, operations = [], 0
    for channel in channels:
        if not isinstance(channel, Channel):
            raise ValueError("each composed item must be a Channel")
        channel = channel.compress(max_states=max_states)
        work = len(current.p) * len(channel.p)
        if operations + work > max_pair_operations:
            raise ValueError("composition exceeds max_pair_operations; no partial answer returned")
        pairs = ((p * r, q * s) for p, q in zip(current.p, current.q) for r, s in zip(channel.p, channel.q))
        current = _group(pairs, max_states)
        operations += work
        sizes.append(len(current.p))
    return Composition(current, len(sizes), tuple(sizes), operations)


def randomized_response(truth_probability=Fraction(3, 4)):
    """A binary channel reporting the true bit with the specified probability."""
    probability = _fraction(truth_probability, "truth_probability", lower=Fraction(1, 2), upper=1)
    return Channel((probability, 1 - probability), (1 - probability, probability))
