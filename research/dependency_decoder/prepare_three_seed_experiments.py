"""Prepare and audit paired 3s5z MAT experiments. Never launches training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

from mat.config import get_config
from research.dependency_decoder.checkpoint_manifest import build_training_manifest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "mat/scripts/train_smac_research.sh"
EVALUATOR = ROOT / "research/dependency_decoder/evaluate_frozen_smac.py"
PYTHON = "/home/sgg/miniconda3/envs/new_titans/bin/python"
MANAGER_MODULE = "research.dependency_decoder.prepare_three_seed_experiments"
MODES = {
    "identity": ("identity", False),
    "dependency": ("dependency", False),
    "dependency-NoBias": ("dependency", True),
    "learned": ("learned", False),
    "obs_norm": ("obs_norm", False),
}
SEEDS = (1, 2, 3)
STEPS = 2_000_000
ROLLOUT_THREADS = 32
EPISODE_LENGTH = 100
PPO_EPOCH = 10
CLIP = "0.05"
INTERVAL = 100
EVAL_EPISODES = 32
BATCH_STEPS = ROLLOUT_THREADS * EPISODE_LENGTH
UPDATES = STEPS // BATCH_STEPS
FINAL_UPDATE_INDEX = UPDATES - 1


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_hashes():
    names = git_output("ls-files").splitlines()
    relevant = (name for name in names if
                (name.startswith("mat/") or name.startswith("research/dependency_decoder/")
                 or name.startswith("tests/")) and name.endswith((".py", ".sh")))
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in relevant}


def launch_env(run_dir, mode, seed):
    code_mode, no_bias = MODES[mode]
    tag = "dg_3s5z_pilot2m_nobias" if no_bias else "dg_3s5z_pilot2m"
    return {
        "DRY_RUN": "1", "AGENT_ORDER_MODE": code_mode,
        "DISABLE_ORDER_ATTENTION_BIAS": "1" if no_bias else "0",
        "STORE_AGENT_ORDERS": "1" if code_mode in ("dependency", "learned") else "0",
        "NUM_ENV_STEPS": str(STEPS), "N_ROLLOUT_THREADS": str(ROLLOUT_THREADS),
        "N_TRAINING_THREADS": "16", "N_EVAL_ROLLOUT_THREADS": "1",
        "PPO_EPOCH": str(PPO_EPOCH), "CLIP_PARAM": CLIP,
        "SAVE_INTERVAL_UPDATES": str(INTERVAL),
        "EVAL_INTERVAL_UPDATES": str(INTERVAL), "EVAL_EPISODES": str(EVAL_EPISODES),
        "USE_EVAL": "1", "ORDER_HIDDEN_DIM": "64",
        "ORDER_TEMPERATURE": "1.0", "ORDER_LOSS_COEF": "0.1",
        "ORDER_PAIR_RANK": "16", "ORDER_GRAPH_SCALE": "0.5",
        "EXP_TAG": tag, "LOG_METHOD_NAME": "MAT_" + mode.replace("-", "_"),
        "MAT_DECODER_LOG_DIR": str(run_dir / "logs"),
        "MAT_DECODER_RESULTS_ROOT": str(run_dir / "training_results"),
    }


def dry_run(run_dir, mode, seed):
    env = dict(os.environ, **launch_env(run_dir, mode, seed))
    proc = subprocess.run(["bash", str(LAUNCHER), "3s5z", "0", str(seed)],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
    output = proc.stdout + proc.stderr
    if proc.returncode or "DRY_RUN=1: command printed; no training started." not in output:
        raise RuntimeError(f"launcher dry run failed: {mode} seed {seed}: {output}")
    line = next((line for line in proc.stdout.splitlines() if line.startswith("command: ")), None)
    if line is None:
        raise RuntimeError("launcher omitted its command")
    words = shlex.split(line.removeprefix("command: "))
    args = words[3:]  # CUDA_VISIBLE_DEVICES=0 python -u train/train_smac.py
    required = {"--agent_order_mode": MODES[mode][0], "--seed": str(seed),
                "--num_env_steps": str(STEPS), "--n_rollout_threads": str(ROLLOUT_THREADS),
                "--ppo_epoch": str(PPO_EPOCH), "--clip_param": CLIP,
                "--save_interval": str(INTERVAL), "--eval_interval": str(INTERVAL),
                "--eval_episodes": str(EVAL_EPISODES),
                "--run_dir": str(run_dir / "training_results")}
    for flag, expected in required.items():
        if flag not in args or args[args.index(flag) + 1] != expected:
            raise RuntimeError(f"dry-run option mismatch: {flag} expected {expected}")
    if ("--disable_order_attention_bias" in args) != MODES[mode][1]:
        raise RuntimeError("NoBias flag mismatch")
    if ("--store_agent_orders" in args) != (MODES[mode][0] in ("dependency", "learned")):
        raise RuntimeError("saved-order flag mismatch")
    return output, args


def expected_manifest(args, sha):
    config, _ = get_config().parse_known_args(args)
    config.map_name = "3s5z"
    return build_training_manifest(config, sha)


def script_environment(env):
    return "\n".join("export " + key + "=" + shlex.quote(value)
                     for key, value in env.items() if key != "DRY_RUN")


def prepare(suite):
    suite = Path(suite).expanduser().resolve()
    if suite.exists():
        raise FileExistsError(f"suite already exists: {suite}")
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked source must be committed before preparing a paired suite")
    sha = git_output("rev-parse", "HEAD")
    hashes = source_hashes()
    suite.mkdir(parents=True)
    write_json(suite / "source_hashes.json", hashes)
    with tarfile.open(suite / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in hashes:
            archive.add(ROOT / name, arcname=name)
    runs = []
    for seed in SEEDS:
        for mode in MODES:
            run_dir = suite / f"3s5z_{mode}_seed{seed}"
            run_dir.mkdir()
            for sub in ("logs", "evaluation", "analysis"):
                (run_dir / sub).mkdir()
            output, args = dry_run(run_dir, mode, seed)
            (run_dir / "logs/dry_run.txt").write_text(output)
            experiment_name = args[args.index("--experiment_name") + 1]
            native_run = (run_dir / "training_results/StarCraft2/3s5z/mat" /
                          experiment_name / "run1")
            final_checkpoint = native_run / f"models/transformer_{FINAL_UPDATE_INDEX}.pt"
            train_manifest = native_run / "mat_decoder_manifest.json"
            eval_result = run_dir / "evaluation/final_32_episodes.json"
            eval_command = [PYTHON, "-u",
                            "-m", "research.dependency_decoder.evaluate_frozen_smac",
                            "--checkpoint", str(final_checkpoint),
                            "--map-name", "3s5z", "--seed", str(seed),
                            "--eval-episodes", str(EVAL_EPISODES),
                            "--agent-order-mode", MODES[mode][0],
                            "--manifest", str(train_manifest),
                            "--output", str(eval_result)]
            env = launch_env(run_dir, mode, seed)
            manifest = {
                "schema_version": 1, "status": "prepared_not_started",
                "role": "primary" if mode in ("identity", "dependency") else "ablation",
                "map": "3s5z", "mode": mode, "training_mode": MODES[mode][0],
                "attention_bias_disabled": MODES[mode][1], "seed": seed,
                "training_source_sha": sha, "source_hashes": str(suite / "source_hashes.json"),
                "requested_env_steps": STEPS, "effective_env_steps": UPDATES * BATCH_STEPS,
                "updates": UPDATES, "ppo_epoch": PPO_EPOCH, "clip_param": float(CLIP),
                "rollout_threads": ROLLOUT_THREADS, "checkpoint_interval_updates": INTERVAL,
                "eval_interval_updates": INTERVAL, "eval_episodes": EVAL_EPISODES,
                "evaluation_seed": seed * 50_000,
                "expected_training_manifest": expected_manifest(args, sha),
                "paths": {"run_directory": str(run_dir), "training_results_root": str(run_dir / "training_results"),
                          "native_run": str(native_run), "training_manifest": str(train_manifest),
                          "training_log_directory": str(run_dir / "logs"),
                          "training_log_glob": str(run_dir / "logs/MAT_*.log"),
                          "tensorboard_directory": str(native_run / "logs"),
                          "checkpoint_directory": str(native_run / "models"),
                          "final_checkpoint": str(final_checkpoint),
                          "native_summary": str(native_run / "logs/summary.json"),
                          "evaluation_plan": str(run_dir / "evaluation_plan.json"),
                          "evaluation_result": str(eval_result)},
                "training_command": ["bash", str(LAUNCHER), "3s5z", "0", str(seed)],
                "training_environment": {k: v for k, v in env.items() if k != "DRY_RUN"},
                "frozen_evaluation_cwd": str(ROOT),
                "frozen_evaluation_command": eval_command,
            }
            write_json(run_dir / "manifest.json", manifest)
            write_json(run_dir / "evaluation_plan.json", {
                "checkpoint_selection": "final fixed-budget update, never best validation score",
                "checkpoint_update_index": FINAL_UPDATE_INDEX,
                "native_smac_episodes": EVAL_EPISODES,
                "seed": seed, "actual_smac_seed": seed * 50_000,
                "deterministic_actions_and_order": True,
                "checkpoint": str(final_checkpoint),
                "training_manifest": str(train_manifest),
                "command": eval_command, "result": str(eval_result)})
            manager = shlex.join([PYTHON, "-m", MANAGER_MODULE])
            quoted_manifest = shlex.quote(str(run_dir / "manifest.json"))
            quoted_suite = shlex.quote(str(suite))
            train_script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                            + "# Prepared only. Execute after explicit GPU0 approval.\n"
                            + "cd " + shlex.quote(str(ROOT)) + "\n"
                            + manager + " verify-run " + quoted_manifest + "\n"
                            + "exec 9>/tmp/hk-mat-gpu0.lock\nflock -n 9 || { echo 'GPU0 lock busy' >&2; exit 1; }\n"
                            + "trap '" + manager + " index " + quoted_suite + "' EXIT\n"
                            + script_environment(env) + "\n"
                            + "DRY_RUN=0 bash " + shlex.quote(str(LAUNCHER))
                            + " 3s5z 0 " + str(seed) + "\n")
            (run_dir / "train.sh").write_text(train_script)
            eval_script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                           + "test -f " + shlex.quote(str(final_checkpoint)) + "\n"
                           + "test -f " + shlex.quote(str(train_manifest)) + "\n"
                           + "exec 9>/tmp/hk-mat-gpu0.lock\nflock -n 9 || { echo 'GPU0 lock busy' >&2; exit 1; }\n"
                           + "cd " + shlex.quote(str(ROOT)) + "\n"
                           + "CUDA_VISIBLE_DEVICES=0 " + shlex.join(eval_command) + "\n"
                           + manager + " index " + quoted_suite + "\n")
            (run_dir / "evaluate_final.sh").write_text(eval_script)
            runs.append(manifest)
    write_json(suite / "suite_manifest.json", {
        "schema_version": 1, "training_source_sha": sha, "map": "3s5z",
        "phase": "pilot", "requested_env_steps_per_run": STEPS,
        "effective_env_steps_per_run": UPDATES * BATCH_STEPS,
        "modes": list(MODES), "seeds": list(SEEDS),
        "primary_run_count": 6, "ablation_run_count": 9,
        "total_runs": len(runs), "run_manifests": [str(Path(m["paths"]["run_directory"]) / "manifest.json")
                                              for m in runs],
        "source_snapshot": str(suite / "source_snapshot.tar.gz"),
        "source_hashes": str(suite / "source_hashes.json")})
    index(suite)
    return runs


def verify_run(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    suite = Path(manifest_path).resolve().parent.parent
    if git_output("rev-parse", "HEAD") != manifest["training_source_sha"]:
        raise RuntimeError("training source SHA changed")
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked training source is dirty")
    hashes = json.loads((suite / "source_hashes.json").read_text())
    for name, expected in hashes.items():
        path = ROOT / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source changed: {name}")
    if Path(manifest["paths"]["native_run"]).parent.exists():
        raise RuntimeError("experiment output already exists; refuse to create run2")
    free = shutil.disk_usage(suite).free
    if free < 25 * 1024 ** 3:
        raise RuntimeError(f"only {free / 1024**3:.1f} GiB free at suite; need at least 25 GiB")
    return True


def index(suite):
    suite = Path(suite).resolve()
    entries = []
    for path in sorted(suite.glob("3s5z_*/manifest.json")):
        data = json.loads(path.read_text())
        p = data["paths"]
        checkpoints = sorted(Path(p["checkpoint_directory"]).glob("transformer_*.pt"))
        training_logs = sorted(Path(p["training_log_directory"]).glob("MAT_*.log"))
        eval_path = Path(p["evaluation_result"])
        runtime_manifest = Path(p["training_manifest"])
        entry = {
            "mode": data["mode"], "seed": data["seed"], "role": data["role"],
            "manifest": str(path), "status": ("evaluated" if eval_path.is_file()
                else "final_checkpoint_ready" if Path(p["final_checkpoint"]).is_file()
                else "started" if runtime_manifest.is_file() or training_logs
                else "prepared_not_started"),
            "training_logs": [str(x) for x in training_logs],
            "checkpoint_count": len(checkpoints),
            "checkpoints": [str(x) for x in checkpoints],
            "final_checkpoint": p["final_checkpoint"],
            "final_checkpoint_exists": Path(p["final_checkpoint"]).is_file(),
            "runtime_training_manifest": str(runtime_manifest) if runtime_manifest.is_file() else None,
            "native_summary": p["native_summary"] if Path(p["native_summary"]).is_file() else None,
            "evaluation_result": str(eval_path) if eval_path.is_file() else None,
        }
        entries.append(entry)
        write_json(path.parent / "results_index.json", entry)
    write_json(suite / "results_index.json", {"run_count": len(entries), "runs": entries})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("suite")
    sub.add_parser("verify-run").add_argument("manifest")
    sub.add_parser("index").add_argument("suite")
    args = parser.parse_args()
    if args.action == "prepare":
        print(f"prepared {len(prepare(args.suite))} runs; training not started")
    elif args.action == "verify-run":
        verify_run(args.manifest)
        print("source, output paths, and disk ready")
    else:
        print(f"indexed {len(index(args.suite))} runs")


if __name__ == "__main__":
    main()
