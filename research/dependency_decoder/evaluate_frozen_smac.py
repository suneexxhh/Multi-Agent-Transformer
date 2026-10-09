"""Read-only deterministic SMAC evaluation of a frozen MAT checkpoint.

Uses the ORIGINAL SMACRunner.eval implementation, original MATPolicy.act,
original StarCraftII eval environment and initial seeds. Unlike a training
script this NEVER calls runner.run(), runner.train(), or PPO updates.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile


def parse_native_eval_win_rate(output, episodes):
    """Extract exactly one official runner evaluation result from its log."""
    values = re.findall(r"eval win rate is ([0-9]*\.?[0-9]+)\.", output)
    if len(values) != 1 or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("Expected one native SMAC eval win rate and a positive episode count")
    rate = float(values[0])
    wins = round(rate * episodes)
    if not 0 <= rate <= 1 or abs(wins / episodes - rate) > 1e-8:
        raise ValueError("SMAC win rate must correspond to an integer win count")
    return int(wins), rate


def evaluate_frozen_checkpoint(checkpoint, map_name="3m", seed=1,
                               n_eval_episodes=32, gpu=True,
                               agent_order_mode="obs_norm"):
    """Run one bounded, native deterministic SMAC evaluation; no training."""
    # Reject invalid calls before importing StarCraftII or constructing
    # any process. This also makes CPU CI independent of PySC2 runtime.
    if not isinstance(n_eval_episodes, int) or not 1 <= n_eval_episodes <= 256:
        raise ValueError("eval episodes must be integer 1..256")
    if not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("invalid evaluation seed")
    if map_name not in ("3m", "3s5z", "3s5z_vs_3s6z", "6h_vs_8z"):
        raise ValueError("frozen research evaluator allows only reviewed SMAC benchmark maps")
    if agent_order_mode not in ("identity", "obs_norm", "random_fixed"):
        raise ValueError("unrecognized MAT agent ordering mode")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)

    import numpy as np
    import torch
    from mat.config import get_config
    from mat.scripts.train.train_smac import parse_args, make_eval_env
    from mat.envs.starcraft2.smac_maps import get_map_params
    from mat.runner.shared.smac_runner import SMACRunner

    args = parse_args([
        "--env_name", "StarCraft2", "--algorithm_name", "mat",
        "--experiment_name", "offline_frozen_eval",
        "--map_name", map_name, "--eval_map_name", map_name,
        "--agent_order_mode", agent_order_mode,
        "--n_rollout_threads", "1", "--n_eval_rollout_threads", "1",
        "--n_training_threads", "2",
        "--episode_length", "100", "--num_env_steps", "100",
        "--eval_episodes", str(n_eval_episodes),
        "--seed", str(seed), "--n_block", "1",
        "--n_embd", "64", "--n_head", "1", "--use_eval",
    ], get_config())
    args.use_eval = True
    args.use_wandb = False
    args.model_dir = None
    args.store_agent_orders = False
    # Eval scenario seed follows MAT make_eval_env and is identical
    # across checkpoint stages for fixed seed and map.
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory(prefix="frozen_mat_smac_eval_") as temp_dir:
        eval_env = None
        runner = None
        try:
            eval_env = make_eval_env(args)
            runner = SMACRunner({
                "all_args": args,
                "envs": eval_env,
                "eval_envs": eval_env,
                "num_agents": get_map_params(map_name)["n_agents"],
                "device": device,
                "run_dir": Path(temp_dir),
            })
            weights = torch.load(str(checkpoint), map_location=device,
                                 weights_only=True)
            runner.policy.transformer.load_state_dict(weights, strict=True)
            runner.policy.transformer.eval()
            log_output = io.StringIO()
            with torch.no_grad(), redirect_stdout(log_output):
                # Calls only the existing MAT eval() method. It does not
                # interact with PPO buffer or invoke any optimizer.
                runner.eval(total_num_steps=0)
            wins, rate = parse_native_eval_win_rate(
                log_output.getvalue(), n_eval_episodes)
            return dict(
                frozen_checkpoint=str(checkpoint), map=map_name,
                evaluation_seed=seed, algorithm=f"MAT {agent_order_mode}",
                deterministic=True, evaluation_episodes=n_eval_episodes,
                wins=wins, win_rate=rate, device=str(device),
                evaluation_only=True, optimizer_updates=0,
                caution=("This is a single-seed, bounded deterministic "
                         "SMAC evaluation, NOT a five-seed benchmark "
                         "or evidence that decoder dependency ordering improves performance."),
            )
        finally:
            if runner is not None and getattr(runner, "writter", None) is not None:
                runner.writter.close()
            if eval_env is not None:
                eval_env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--map-name", default="3m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--agent-order-mode", choices=("identity","obs_norm","random_fixed"),
                        default="obs_norm")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_frozen_checkpoint(
        args.checkpoint, args.map_name, args.seed,
        args.eval_episodes, not args.cpu, args.agent_order_mode)
    txt = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
