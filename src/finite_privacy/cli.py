"""Audit one specified neighboring pair of finite output laws from JSON."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from .core import Channel, compose_independent, exact_json


def _rational(value):
    if type(value) not in (str, int):
        raise ValueError("probabilities must be integer or rational/decimal strings")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rational value") from exc


def _argument(value):
    try:
        return _rational(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"duplicate JSON field {k!r}")
        result[k] = v
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channel", type=Path)
    parser.add_argument("--releases", type=int, default=1)
    parser.add_argument("--alpha", type=_argument, default=Fraction(1))
    parser.add_argument("--target-delta", type=_argument, default=Fraction(1, 100))
    args = parser.parse_args()
    try:
        if not 0 <= args.releases <= 100:
            raise ValueError("releases must be in [0,100]")
        data = json.loads(args.channel.read_text(encoding="utf-8"), object_pairs_hook=_unique)
        if not isinstance(data, dict) or set(data) != {"p", "q"}:
            raise ValueError("channel JSON must contain exactly p and q")
        if not isinstance(data["p"], list) or not isinstance(data["q"], list):
            raise ValueError("p and q must be arrays")
        channel = Channel(tuple(map(_rational, data["p"])), tuple(map(_rational, data["q"])))
        composed = compose_independent(channel for _ in range(args.releases))
        report = {"scope": "One specified neighboring pair; releases assumed conditionally independent.",
                  "releases": args.releases, "compressed_states": len(composed.channel.p),
                  "states_per_step": composed.states_per_step, "pair_operations": composed.pair_operations,
                  "profile": composed.channel.profile(args.alpha).to_dict(),
                  "target_delta": args.target_delta,
                  "minimum_alpha": composed.channel.calibrate(args.target_delta),
                  "delta_floor": composed.channel.delta_floor,
                  "ratio_bucket_channel": composed.channel.to_dict()}
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(exact_json(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
