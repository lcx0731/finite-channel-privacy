"""Synthetic, exact channel experiments; no private data or release mechanism."""

import argparse
from fractions import Fraction as F
from itertools import product
import json
from math import log
from pathlib import Path

from .core import Channel, compose_independent, exact_json, randomized_response
from .reference import enumerate_transcripts, exhaustive_alpha, exhaustive_events


def shared_coin_channels():
    # Outputs ordered 00,01,10,11. U=R and V=R xor secret, R is one shared fair bit.
    joint = Channel((F(1, 2), 0, 0, F(1, 2)), (0, F(1, 2), F(1, 2), 0))
    first = joint.postprocess(((1, 0), (1, 0), (0, 1), (0, 1)))
    second = joint.postprocess(((1, 0), (0, 1), (1, 0), (0, 1)))
    return joint, first, second


def run_demo():
    rr = randomized_response(F(3, 4))
    profiles, calibration = [], []
    for n in (1, 4, 12):
        composition = compose_independent(rr for _ in range(n))
        # Full transcript enumeration is separate from the ratio-bucket algorithm.
        raw_p, raw_q = enumerate_transcripts([rr] * n)
        alphas = (F(1), F(3), F(9), F(27), F(81), F(3**n))
        for alpha in alphas:
            expected_f = sum((max(p - alpha * q, F()) for p, q in zip(raw_p, raw_q)), F())
            expected_r = sum((max(q - alpha * p, F()) for p, q in zip(raw_p, raw_q)), F())
            actual = composition.channel.profile(alpha)
            assert (actual.forward_delta, actual.reverse_delta) == (expected_f, expected_r)
        minimum = composition.channel.calibrate(F(1, 100))
        assert composition.channel.profile(minimum).delta <= F(1, 100)
        assert composition.channel.calibrate(0) == 3**n
        calibration.append({"releases": n, "raw_transcripts": 2**n,
                            "ratio_states": len(composition.channel.p),
                            "pair_operations": composition.pair_operations,
                            "delta_at_alpha_1": composition.channel.profile(1).delta,
                            "delta_at_alpha_3": composition.channel.profile(3).delta,
                            "target_delta": F(1, 100), "minimum_alpha": minimum,
                            "pure_alpha": F(3**n), "transcript_checks_passed": True})
        grid = [F(11, 10)**i for i in range(151)]
        profiles.append({"releases": n, "alpha": grid,
                         "delta": [composition.channel.profile(a).delta for a in grid]})
    raw3 = Channel(*enumerate_transcripts([rr] * 3))
    majority_kernel = tuple((int(sum(bits) < 2), int(sum(bits) >= 2)) for bits in product((0, 1), repeat=3))
    majority = raw3.postprocess(majority_kernel)
    for alpha in (F(1), F(3), F(9), F(27)):
        assert majority.profile(alpha).delta <= raw3.profile(alpha).delta
        assert exhaustive_events(raw3.p, raw3.q, alpha) == (raw3.profile(alpha).forward_delta, raw3.profile(alpha).reverse_delta)
    assert exhaustive_alpha(raw3.p, raw3.q, F(1, 100)) == raw3.calibrate(F(1, 100))
    joint, first, second = shared_coin_channels()
    independent = compose_independent((first, second)).channel
    assert first.profile(1).delta == second.profile(1).delta == independent.profile(1).delta == 0
    assert all(joint.profile(a).delta == 1 for a in (1, 3, 10**6))
    assert joint.calibrate(F(99, 100)) is None
    return exact_json({
        "scope": "Exact rational laws for one binary neighboring pair; all examples are synthetic.",
        "randomized_response": rr.to_dict(), "calibration": calibration, "profile_grid": profiles,
        "postprocessing": {"raw_three_release_pure_alpha": raw3.calibrate(0),
                           "majority_pure_alpha": majority.calibrate(0),
                           "raw_delta_at_9": raw3.profile(9).delta,
                           "majority_delta_at_9": majority.profile(9).delta,
                           "majority_channel": majority.to_dict()},
        "shared_coin_counterexample": {
            "outputs": ["00", "01", "10", "11"], "joint_channel": joint.to_dict(),
            "first_marginal": first.to_dict(), "second_marginal": second.to_dict(),
            "individual_delta_at_1": F(0), "independent_product_delta_at_1": independent.profile(1).delta,
            "actual_joint_delta_at_1": joint.profile(1).delta, "actual_joint_delta_floor": joint.delta_floor,
            "minimum_alpha_for_delta_099": joint.calibrate(F(99, 100)),
            "explanation": "The second release conditional on the first reveals the secret; marginal guarantees do not justify independent composition.",
        },
        "reference_checks": {"full_transcript_sizes": [2, 16, 4096], "three_release_all_event_subsets": 256,
                             "exhaustive_event_calibration_passed": True},
    })


def write_outputs(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = run_demo()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "rr-channel.json").write_text(json.dumps(summary["randomized_response"], indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout="constrained")
    for curve, color in zip(summary["profile_grid"], ("#26738b", "#d17932", "#61478e")):
        # Floats are for plotting only; the stored alpha/delta values are exact.
        x = [log(float(F(a))) for a in curve["alpha"]]
        y = [float(F(d)) for d in curve["delta"]]
        axes[0].plot(x, y, color=color, label=f'{curve["releases"]} independent release(s)')
    axes[0].set(title="Exact profiles: randomized response (p = 3/4)", xlabel="epsilon = log(alpha), display approximation",
                ylabel="Tight two-direction delta", ylim=(-0.03, 1.04))
    axes[0].legend(frameon=False)
    labels = ["U alone", "V alone", "Independent\nmodel", "Actual joint\n(U, V)"]
    axes[1].bar(labels, [0, 0, 0, 1], color=["#26738b"] * 3 + ["#bc4d42"], width=0.58)
    axes[1].scatter(range(3), [0, 0, 0], color="#26738b", s=30, zorder=5)
    for i, d in enumerate([0, 0, 0, 1]):
        axes[1].text(i, d + 0.035, str(d), ha="center", va="bottom")
    axes[1].set(title="Shared coin: U = R, V = R xor secret", ylabel="Delta at epsilon = 0", ylim=(-0.03, 1.17))
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "privacy-profiles.png", dpi=170, metadata={"Software": "finite-channel-privacy"})
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    summary = write_outputs(args.output)
    print(json.dumps({"calibration": summary["calibration"], "postprocessing": summary["postprocessing"],
                      "shared_coin_counterexample": summary["shared_coin_counterexample"]}, indent=2))


if __name__ == "__main__":
    main()
