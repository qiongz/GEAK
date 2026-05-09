"""Shared Triton-Gluon documentation profile metadata."""

from __future__ import annotations

MANDATORY_GLUON_DOC_KEYS: tuple[str, ...] = (
    "gluon_skill_path",
    "gluon_always_read_path",
    "gluon_search_policies_path",
)

GLUON_DOC_PROFILE_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "extension_l0_minimal": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_component_traits_path",
        "gluon_api_reference_path",
    ),
    "nv_to_amd_translation": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_component_traits_path",
        "gluon_architecture_notes_path",
        "gluon_real_patterns_path",
        "gluon_api_reference_path",
    ),
    "memory_lowering": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_component_traits_path",
        "gluon_api_reference_path",
    ),
    "matrix_lowering": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_component_traits_path",
        "gluon_architecture_notes_path",
        "gluon_api_reference_path",
    ),
    "shape_bucketed_dispatch": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_component_traits_path",
        "gluon_real_patterns_path",
    ),
    "jit_aot_sensitive": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_architecture_notes_path",
        "gluon_api_reference_path",
    ),
    "shared_transplant": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_component_traits_path",
        "gluon_real_patterns_path",
    ),
    "gluon_variant_from_anchor": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_component_traits_path",
        "gluon_api_reference_path",
        "gluon_real_patterns_path",
    ),
    "hybrid_dispatch": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_real_patterns_path",
    ),
    "hybrid_dispatch_from_evidence": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_real_patterns_path",
    ),
    "base_or_shared_gluon": (
        *MANDATORY_GLUON_DOC_KEYS,
        "gluon_component_traits_path",
    ),
}

GLUON_DOC_PROFILE_VALUES: tuple[str, ...] = tuple(GLUON_DOC_PROFILE_REQUIRED_KEYS)


def add_unique_doc_key(keys: list[str], key: str) -> None:
    """Append a documentation metadata key once."""
    if key and key not in keys:
        keys.append(key)


def required_doc_keys_for_profile(profile: str | None) -> list[str]:
    """Return the deterministic documentation keys for a Gluon doc profile."""
    normalized = str(profile or "").strip().lower()
    return list(GLUON_DOC_PROFILE_REQUIRED_KEYS.get(normalized, MANDATORY_GLUON_DOC_KEYS))
