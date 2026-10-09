"""Read-only quality review of multi-stage MAT Decoder diagnostic JSON.

Outputs descriptive stage changes, never causal edges or training labels.
No torch, SMAC startup, GPU allocation, model updates, or PPO changes.
"""
import argparse
import json
import math
from pathlib import Path


def review_stage_summary(summary, noise_floor_tv=None):
    """Assess evidence limitations without making up a significance threshold.

    Input is the private intermediate pilot's stage_summary.json schema.
    The optional noise floor must be measured independently (e.g. from a
    numerical-null control), not tuned on the same observed edges.
    Even a TV value above that floor is NOT automatically a valid label.
    """
    if not isinstance(summary, dict) or not isinstance(summary.get("stages"), list):
        raise ValueError("expected intermediate diagnostic stage summary")
    stages = summary["stages"]
    if not stages:
        raise ValueError("stages must be nonempty")
    if noise_floor_tv is not None and (
            not isinstance(noise_floor_tv, (int, float)) or
            not math.isfinite(noise_floor_tv) or noise_floor_tv <= 0):
        raise ValueError("noise_floor_tv must be a finite positive independent control")

    inspected = []
    seen = set()
    for stage in stages:
        index = stage.get("episode_index")
        measured = stage.get("observable_pairs")
        total = stage.get("total_directed_pairs")
        counts = stage.get("context_pair_counts")
        kl, tv = stage.get("mean_KL"), stage.get("mean_TV")
        if (not isinstance(index, int) or index < 0 or index in seen or
                not isinstance(measured, int) or not isinstance(total, int) or
                total <= 0 or measured < 0 or measured > total or
                not isinstance(counts, list) or not counts or
                any(not isinstance(v, int) or v < 0 for v in counts) or
                sum(counts) != measured):
            raise ValueError("invalid stage provenance, coverage or pair counts")
        if measured == 0:
            if kl is not None or tv is not None:
                raise ValueError("unmeasured pairs cannot have mean KL/TV")
        elif (not isinstance(kl, (int, float)) or
              not isinstance(tv, (int, float)) or
              not math.isfinite(kl) or not math.isfinite(tv) or
              kl < 0 or not 0 <= tv <= 1):
            raise ValueError("finite KL and bounded TV required for measured pairs")
        seen.add(index)
        inspected.append(dict(
            episode_index=index,
            train_steps_at_capture=stage.get("train_steps_at_capture"),
            observed_directed_pairs=measured,
            possible_directed_pairs=total,
            observable_fraction=measured / total,
            completely_unidentifiable_contexts=sum(v == 0 for v in counts),
            mean_KL=kl,
            mean_TV=tv,
            tv_over_external_noise_floor=(
                (tv / noise_floor_tv)
                if tv is not None and noise_floor_tv is not None else None),
        ))
    inspected.sort(key=lambda item: item["episode_index"])
    # Stages sample DISTINCT observations under different checkpoints.
    # Stage-level KL ratios cannot isolate learning-induced parameter effects.
    earliest, latest = inspected[0], inspected[-1]
    delta = (latest["mean_TV"] - earliest["mean_TV"]
             if earliest["mean_TV"] is not None and
                latest["mean_TV"] is not None else None)
    reasons = []
    if summary.get("no_eval_win_rate", True):
        reasons.append("No independent policy win-rate evaluation")
    reasons.append("Only one documented experiment seed")
    reasons.append("Four temporally adjacent observations per stage are not independent trajectories")
    reasons.append("Only one saved decoder permutation; reverse-order edges are unidentifiable")
    if noise_floor_tv is None:
        reasons.append("No independently calibrated policy-score noise floor")
    else:
        reasons.append("TV above a noise floor is not a causal or precedence ground-truth label")
    return dict(
        method="offline descriptive diagnostics, NOT supervised dependency labels",
        map=summary.get("map"),
        seed=summary.get("seed"),
        stage_count=len(inspected),
        stages=inspected,
        raw_last_minus_first_mean_TV=delta,
        stage_differences_confounded_by_observation_change=True,
        allowed_for_precedence_label_training=False,
        reasons=reasons,
        disclaimer=("A single autoregressive action order, single seed and "
                    "adjacent observations cannot identify true directed "
                    "multi-agent dependence or prove policy improvement."),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--external-tv-noise-floor", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    source = json.loads(args.summary.read_text(encoding="utf-8"))
    review = review_stage_summary(source, args.external_tv_noise_floor)
    raw = json.dumps(review, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw + "\n", encoding="utf-8")
    print(raw)
    return review


if __name__ == "__main__":
    main()
