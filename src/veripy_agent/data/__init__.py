from veripy_agent.contracts import TaskSpec

from .curriculum import (
    admit_curriculum_candidates,
    build_curriculum_candidates,
    curriculum_manifest,
)
from .generator import (
    FAMILIES,
    GeneratedTask,
    generate_tasks,
    manifest,
    validate_split_isolation,
    validate_task_bundle,
    write_tasks,
)
from .public_adapters import (
    ADAPTERS,
    PublicExample,
    audit_cross_split_contamination,
    audit_examples,
    materialize_public_examples,
    public_task_spec,
)
from .public_sources import admitted_training_sources, load_source_lock, load_source_registry

__all__ = [
    "FAMILIES",
    "GeneratedTask",
    "ADAPTERS",
    "PublicExample",
    "TaskSpec",
    "admitted_training_sources",
    "audit_cross_split_contamination",
    "audit_examples",
    "admit_curriculum_candidates",
    "build_curriculum_candidates",
    "curriculum_manifest",
    "generate_tasks",
    "manifest",
    "materialize_public_examples",
    "load_source_lock",
    "load_source_registry",
    "public_task_spec",
    "validate_split_isolation",
    "validate_task_bundle",
    "write_tasks",
]
