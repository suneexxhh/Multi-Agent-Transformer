"""Versioned training metadata required for reproducible frozen MAT evaluation.

Only Python stdlib dependencies: safe to validate manifests in hosted CPU CI.
Checkpoints in this project contain model STATE_DICT ONLY, not optimizer/RNG
state: the manifest MUST NOT be mistaken for a resumable training checkpoint.
"""
import json
from pathlib import Path
import re
import math

SCHEMA_VERSION = 2
LEGACY_MODEL_DEFAULTS = {
    "adaptive_order_attention_bias": False,
    "order_attention_scale_max": 2.0,
    "order_influence_align_coef": 0.0,
    "order_influence_max_alternatives": 2,
    "order_influence_effect_scale": 1e-5,
    "order_influence_min_effect": 1e-7,
}
MODEL_FIELDS = (
    "env_name", "map_name", "algorithm_name", "agent_order_mode",
    "n_block", "n_embd", "n_head", "order_hidden_dim",
    "order_temperature", "order_pair_rank", "order_graph_scale",
    "order_prefix_scale", "disable_order_attention_bias", "encode_state", "dec_actor",
    "share_actor", "store_agent_orders", "agent_order_seed",
    "adaptive_order_attention_bias", "order_attention_scale_max",
    "order_influence_align_coef", "order_influence_max_alternatives",
    "order_influence_effect_scale", "order_influence_min_effect",
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
    cfg = {name: getattr(all_args, name, LEGACY_MODEL_DEFAULTS.get(name))
           for name in (*MODEL_FIELDS, *PROTOCOL_FIELDS)}
    if cfg["agent_order_mode"] not in ORDER_MODES or cfg["algorithm_name"] != "mat":
        raise ValueError("not a supported MAT Decoder experiment")
    if (cfg["adaptive_order_attention_bias"] and
            (cfg["agent_order_mode"] != "dependency_prefix" or cfg["disable_order_attention_bias"])):
        raise ValueError("adaptive bias requires enabled dependency_prefix bias")
    if (not math.isfinite(cfg["order_attention_scale_max"]) or cfg["order_attention_scale_max"] <= 0 or
            not math.isfinite(cfg["order_influence_align_coef"]) or cfg["order_influence_align_coef"] < 0 or
            not isinstance(cfg["order_influence_max_alternatives"], int) or
            cfg["order_influence_max_alternatives"] < 1 or
            not math.isfinite(cfg["order_influence_effect_scale"]) or cfg["order_influence_effect_scale"] <= 0 or
            not math.isfinite(cfg["order_influence_min_effect"]) or cfg["order_influence_min_effect"] < 0):
        raise ValueError("invalid V3.1 model hyperparameters")
    if cfg["order_influence_align_coef"] and cfg["agent_order_mode"] != "dependency_prefix":
        raise ValueError("influence alignment requires dependency_prefix")
    return dict(schema_version=SCHEMA_VERSION,
                training_source_sha=source_sha, source_dirty=bool(source_dirty),
                checkpoint_format="transformer_state_dict_only",
                resumable_training_state=False,
                config=cfg)


def load_training_manifest(path, map_name, mode):
    if mode not in ORDER_MODES:
        raise ValueError("unknown MAT decoding mode")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") not in (1, SCHEMA_VERSION):
        raise ValueError("unsupported training manifest schema")
    cfg = raw.get("config")
    expected = (tuple(k for k in MODEL_FIELDS if k not in LEGACY_MODEL_DEFAULTS)
                if raw["schema_version"] == 1 else MODEL_FIELDS)
    if not isinstance(cfg, dict) or any(k not in cfg for k in (*expected, *PROTOCOL_FIELDS)):
        raise ValueError("incomplete training manifest")
    if raw["schema_version"] == 1:
        cfg = {**LEGACY_MODEL_DEFAULTS, **cfg}
        raw = {**raw, "config": cfg}
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
    if (cfg["adaptive_order_attention_bias"] and
            (mode != "dependency_prefix" or cfg["disable_order_attention_bias"])):
        raise ValueError("invalid adaptive attention manifest")
    if (not math.isfinite(cfg["order_attention_scale_max"]) or cfg["order_attention_scale_max"] <= 0 or
            not math.isfinite(cfg["order_influence_align_coef"]) or cfg["order_influence_align_coef"] < 0 or
            not isinstance(cfg["order_influence_max_alternatives"], int) or
            cfg["order_influence_max_alternatives"] < 1 or
            not math.isfinite(cfg["order_influence_effect_scale"]) or cfg["order_influence_effect_scale"] <= 0 or
            not math.isfinite(cfg["order_influence_min_effect"]) or cfg["order_influence_min_effect"] < 0):
        raise ValueError("invalid V3.1 checkpoint hyperparameters")
    if cfg["order_influence_align_coef"] and mode != "dependency_prefix":
        raise ValueError("invalid influence alignment manifest")
    return raw


def evaluation_model_flags(manifest):
    """Arguments that reconstruct the *same* MAT architecture and bias gate."""
    cfg = manifest["config"]
    flags = []
    for k in ("n_block", "n_embd", "n_head", "order_hidden_dim",
              "order_temperature", "order_pair_rank", "order_graph_scale",
              "order_prefix_scale", "agent_order_seed", "order_attention_scale_max",
              "order_influence_align_coef", "order_influence_max_alternatives",
              "order_influence_effect_scale", "order_influence_min_effect"):
        flags += ["--" + k, str(cfg[k])]
    for name in ("encode_state", "share_actor", "store_agent_orders",
                 "disable_order_attention_bias", "adaptive_order_attention_bias"):
        if cfg[name]:
            flags.append("--" + name)
    if cfg["dec_actor"]:
        raise ValueError("non-original dec_actor cannot be frozen-evaluated here")
    return flags
