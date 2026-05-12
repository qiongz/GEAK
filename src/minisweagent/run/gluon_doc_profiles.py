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


def task_requires_gluon_worker_docs(
    *,
    kernel_type: str | None,
    required_output: str | None,
    implementation_layer: str | None = None,
    extension_layer: str | None = None,
    doc_profile: str | None = None,
    task_body: str = "",
    label: str | None = None,
) -> bool:
    """Return whether a task should receive worker-side Gluon docs/context."""
    if str(kernel_type or "").strip().lower() != "triton":
        return False

    required = str(required_output or "").strip().lower()
    implementation = str(implementation_layer or "").strip().lower()
    extension = str(extension_layer or "").strip().lower()
    profile = str(doc_profile or "").strip().lower()
    text = "\n".join(
        str(part or "")
        for part in (
            task_body,
            label,
            implementation,
            profile,
        )
    ).lower()
    has_gluon_signal = bool(
        required in {"amd_gluon", "mixed"}
        or "amd_gluon" in implementation
        or "amd-gluon" in implementation
        or any(
            marker in text
            for marker in (
                "composition type: shared_transplant",
                "composition type: gluon_variant",
                "composition type: hybrid_dispatch",
                "shared_transplant",
                "gluon_variant",
                "@gluon.jit",
                "amd gluon overlay",
                "amd_gluon variant",
                "required output dialect: amd_gluon",
                "required_output_dialect=amd_gluon",
            )
        )
    )

    if required in {"amd_gluon", "mixed"}:
        return True
    if "amd_gluon" in implementation or "amd-gluon" in implementation:
        return True
    if extension in {"l0", "l1", "hybrid"} and has_gluon_signal:
        return True
    if profile in {
        "extension_l0_minimal",
        "nv_to_amd_translation",
        "shared_transplant",
        "gluon_variant_from_anchor",
        "hybrid_dispatch",
        "hybrid_dispatch_from_evidence",
    }:
        return True
    return has_gluon_signal
