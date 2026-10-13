"""Fail-closed request parameter checks; not a substitute for server capability probing."""

COMMON = frozenset({
    "temperature", "top_p", "max_tokens", "seed", "stop",
    "frequency_penalty", "presence_penalty", "logit_bias",
})
# These are request keys, not a claim of identical interpretation across engines.
LLAMA = frozenset({
    "top_k", "min_p", "typical_p", "repeat_penalty", "repeat_last_n",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "dry_sequence_breakers", "xtc_probability", "xtc_threshold", "samplers",
    "mirostat", "mirostat_tau", "mirostat_eta", "dynatemp_range",
    "dynatemp_exponent", "n_keep", "cache_prompt",
})
KOBOLD = frozenset({
    "top_k", "min_p", "typical", "typical_p", "rep_pen", "rep_pen_range",
    "rep_pen_slope", "sampler_order", "sampler_seed", "tfs", "top_a",
    "mirostat", "mirostat_tau", "mirostat_eta", "dynatemp_range",
    "dynatemp_exponent", "stop_sequence", "use_default_badwordsids",
})


def validate_sampling(name: str, mode: str, sampling: dict) -> None:
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be a JSON object")
    if name == "mock":
        allowed = COMMON | LLAMA | KOBOLD
    elif name == "llamacpp-native" and mode == "completion":
        allowed = COMMON | LLAMA
    elif name == "koboldcpp-native" and mode == "completion":
        allowed = COMMON | KOBOLD
    else:
        # Do not assume engine-native sampling fields work on OpenAI-compatible routes.
        allowed = COMMON
    unknown = sorted(set(sampling) - allowed)
    if unknown:
        raise ValueError(
            f"Unverified sampling parameter(s) for {name}/{mode}: {', '.join(unknown)}. "
            "Use an engine-native adapter or extend its validated capability profile."
        )
    if "max_tokens" in sampling and (type(sampling["max_tokens"]) is not int or sampling["max_tokens"] <= 0):
        raise ValueError("max_tokens must be a positive integer")
    if "temperature" in sampling and (not isinstance(sampling["temperature"], (int, float)) or isinstance(sampling["temperature"], bool) or sampling["temperature"] < 0):
        raise ValueError("temperature must be a nonnegative number")
    for key in ("top_p", "min_p"):
        if key in sampling and (not isinstance(sampling[key], (int, float)) or isinstance(sampling[key], bool) or not 0 <= sampling[key] <= 1):
            raise ValueError(f"{key} must be between 0 and 1")
