"""Post-result schema-floor amendment; independent human requirements retained."""

from despamo.appearance.provenance import digest

POLICY_VERSION = "qwen-schema98-human-v2"
MINIMUM_SCHEMA_PERCENT = 98
ORIGINAL_SCHEMA_PERCENT = 99
HUMAN_REQUIREMENTS = {
    "minimum_correctness": 0.85,
    "minimum_visibility": 0.90,
    "minimum_side_accuracy": 0.95,
    "maximum_unsupported_biometrics": 0.05,
    "maximum_forbidden": 0,
    "minimum_factor_coverage": 1000,
}


def schema_eligibility(valid: int, total: int) -> dict:
    """Evaluate a complete split using integer arithmetic, not a rounded rate."""
    if type(valid) is not int or type(total) is not int or total <= 0 or not 0 <= valid <= total:
        raise ValueError("invalid complete-split schema counts")
    policy = {
        "policy_version": POLICY_VERSION,
        "amendment": "post-result",
        "minimum_schema_rate": MINIMUM_SCHEMA_PERCENT / 100,
        "original_minimum_schema_rate": ORIGINAL_SCHEMA_PERCENT / 100,
        "human_requirements": dict(HUMAN_REQUIREMENTS),
    }
    minimum = (MINIMUM_SCHEMA_PERCENT * total + 99) // 100
    return {
        **policy,
        "policy_hash": digest(policy),
        "valid_count": valid,
        "failed_count": total - valid,
        "total_count": total,
        "schema_rate": valid / total,
        "minimum_valid_count": minimum,
        "maximum_failed_count": total - minimum,
        "original_minimum_valid_count": (ORIGINAL_SCHEMA_PERCENT * total + 99) // 100,
        "schema_pass": valid * 100 >= MINIMUM_SCHEMA_PERCENT * total,
        "original_schema_pass": valid * 100 >= ORIGINAL_SCHEMA_PERCENT * total,
    }
