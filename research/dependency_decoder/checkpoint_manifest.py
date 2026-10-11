"""Versioned training metadata required for reproducible frozen MAT evaluation.

Only Python stdlib dependencies: safe to validate manifests in hosted CPU CI.
Checkpoints in this project contain model STATE_DICT ONLY, not optimizer/RNG
state: the manifest MUST NOT be mistaken for a resumable training checkpoint.
"""
import json
from pathlib import Path
import re

SCHEMA_VERSION = 1
MODEL_FIELDS = (
    "env_name", "map_name", "algorithm_name", "agent_order_mode",
    "n_block", "n_embd", "n_head", "order_hidden_dim",
    "order_temperature", "order_pair_rank", "order_graph_scale",
    "order_prefix_scale", "disable_order_attention_bias", "encode_state", "dec_actor",
    "share_actor", "store_agent_orders", "agent_order_seed",
)
PROTOCOL_FIELDS = (
    "seed", "n_rollout_threads", "n_eval_rollout_threads",
    "num_env_steps", "episode_length", "ppo_epoch", "clip_param",
    "save_interval", "eval_episodes", "use_eval",
)
ORDER_MODES = frozenset(("identity", "obs_norm", "random_fixed", "learned", "dependency", "dependency_prefix"))


def build_training_manifest(all_args, source_sha, source_dirty=False):
    if not isinstance(source_sha, str) or not re.fullmatch("[0-9a-f]{40}", source_sha):
        raise ValueError("full 40-character training git SHA required")
    cfg = {name: getattr(all_args, name) for name in (*MODEL_FIELDS, *PROTOCOL_FIELDS)}
    if cfg["agent_order_mode"] not in ORDER_MODES or cfg["algorithm_name"] != "mat":
        raise ValueError("not a supported MAT Decoder experiment")
    return dict(schema_version=SCHEMA_VERSION,
                training_source_sha=source_sha, source_dirty=bool(source_dirty),
                checkpoint_format="transformer_state_dict_only",
                resumable_training_state=False,
                config=cfg)


def load_training_manifest(path, map_name, mode):
    if mode not in ORDER_MODES:
        raise ValueError("unknown MAT decoding mode")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training manifest schema")
    cfg = raw.get("config")
    if not isinstance(cfg, dict) or any(k not in cfg for k in (*MODEL_FIELDS, *PROTOCOL_FIELDS)):
        raise ValueError("incomplete training manifest")
    if raw.get("checkpoint_format") != "transformer_state_dict_only":
        raise ValueError("unsupported frozen checkpoint format")
    if (cfg["map_name"] != map_name or cfg["agent_order_mode"] != mode or
            cfg["env_name"] != "StarCraft2" or cfg["algorithm_name"] != "mat"):
        raise ValueError("checkpoint manifest map / ordering / algorithm mismatch")
    if (mode in ("learned", "dependency", "dependency_prefix")) != bool(cfg["store_agent_orders"]):
        # identity or obs_norm may intentionally save their order as well;
        # the requirement applies only one-way to learned policies.
        if mode in ("learned", "dependency", "dependency_prefix"):
            raise ValueError("learned checkpoint must store rollout order")
    if (any(not isinstance(cfg[k], int) or cfg[k] < 1 for k in (
            "n_block", "n_embd", "n_head", "order_hidden_dim", "order_pair_rank"))
            or cfg["n_embd"] % cfg["n_head"] != 0):
        raise ValueError("invalid model dimensions in manifest")
    if not re.fullmatch("[0-9a-f]{40}", str(raw.get("training_source_sha", ""))):
        raise ValueError("missing full training source SHA")
    if raw.get("source_dirty"):
        raise ValueError("cannot certify an evaluation from an uncommitted training source")
    return raw


def evaluation_model_flags(manifest):
    """Arguments that reconstruct the *same* MAT architecture and bias gate."""
    cfg = manifest["config"]
    flags = []
    for k in ("n_block", "n_embd", "n_head", "order_hidden_dim",
              "order_temperature", "order_pair_rank", "order_graph_scale",
              "order_prefix_scale", "agent_order_seed"):
        flags += ["--" + k, str(cfg[k])]
    for name in ("encode_state", "share_actor", "store_agent_orders",
                 "disable_order_attention_bias"):
        if cfg[name]:
            flags.append("--" + name)
    if cfg["dec_actor"]:
        raise ValueError("non-original dec_actor cannot be frozen-evaluated here")
    return flags
