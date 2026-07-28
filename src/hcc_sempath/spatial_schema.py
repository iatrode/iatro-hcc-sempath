from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


CELL_INSTANCE_DENSITY = "cell_instance_density"
CONTINUOUS_AREA = "continuous_area"
PIGMENT_BURDEN = "pigment_burden"
STRUCTURE_INSTANCE_AREA = "structure_instance_area"


@dataclass(frozen=True)
class SpatialComponentSpec:
    """Scientific measurement contract for one spatial component."""

    name: str
    mode: str
    supports_instance_count: bool
    supports_density: bool
    supports_area: bool
    supports_focus_density: bool = False


DEFAULT_SPATIAL_COMPONENT_SPECS = (
    SpatialComponentSpec(
        "hepatocellular-parenchyma",
        CELL_INSTANCE_DENSITY,
        supports_instance_count=True,
        supports_density=True,
        supports_area=False,
    ),
    SpatialComponentSpec(
        "necrosis",
        CONTINUOUS_AREA,
        supports_instance_count=False,
        supports_density=False,
        supports_area=True,
    ),
    SpatialComponentSpec(
        "hemorrhage",
        CELL_INSTANCE_DENSITY,
        supports_instance_count=True,
        supports_density=True,
        supports_area=False,
    ),
    SpatialComponentSpec(
        "bile-pigment",
        PIGMENT_BURDEN,
        supports_instance_count=False,
        supports_density=False,
        supports_area=True,
        supports_focus_density=True,
    ),
    SpatialComponentSpec(
        "inflammatory-cell",
        CELL_INSTANCE_DENSITY,
        supports_instance_count=True,
        supports_density=True,
        supports_area=False,
    ),
    SpatialComponentSpec(
        "fibroblast",
        CELL_INSTANCE_DENSITY,
        supports_instance_count=True,
        supports_density=True,
        supports_area=False,
    ),
    SpatialComponentSpec(
        "fibrous-stroma",
        CONTINUOUS_AREA,
        supports_instance_count=False,
        supports_density=False,
        supports_area=True,
    ),
    SpatialComponentSpec(
        "steatosis-vacuolation",
        STRUCTURE_INSTANCE_AREA,
        supports_instance_count=True,
        supports_density=False,
        supports_area=True,
    ),
    SpatialComponentSpec(
        "small-vessel",
        STRUCTURE_INSTANCE_AREA,
        supports_instance_count=True,
        supports_density=False,
        supports_area=True,
    ),
    SpatialComponentSpec(
        "large-vessel",
        STRUCTURE_INSTANCE_AREA,
        supports_instance_count=True,
        supports_density=False,
        supports_area=True,
    ),
    SpatialComponentSpec(
        "ductular-portal",
        STRUCTURE_INSTANCE_AREA,
        supports_instance_count=True,
        supports_density=False,
        supports_area=True,
    ),
)

DEFAULT_SPATIAL_COMPONENTS = tuple(
    spec.name for spec in DEFAULT_SPATIAL_COMPONENT_SPECS
)
_SPEC_BY_NAME = {
    spec.name: spec for spec in DEFAULT_SPATIAL_COMPONENT_SPECS
}


def spatial_component_specs(
    component_names: Sequence[str],
    *,
    unknown_mode: str | None = None,
) -> tuple[SpatialComponentSpec, ...]:
    """Resolve component contracts while preserving the supplied class order.

    ``unknown_mode`` exists only for small synthetic tests and downstream
    experiments. Production eleven-component manifests are validated against
    ``DEFAULT_SPATIAL_COMPONENTS`` before reaching this function.
    """

    result: list[SpatialComponentSpec] = []
    for name in component_names:
        canonical = str(name)
        spec = _SPEC_BY_NAME.get(canonical)
        if spec is not None:
            result.append(spec)
            continue
        if unknown_mode == CELL_INSTANCE_DENSITY:
            result.append(
                SpatialComponentSpec(
                    canonical,
                    CELL_INSTANCE_DENSITY,
                    supports_instance_count=True,
                    supports_density=True,
                    supports_area=False,
                )
            )
            continue
        raise ValueError(f"unknown spatial component contract: {canonical!r}")
    return tuple(result)


def spatial_component_metadata(
    component_names: Sequence[str],
) -> list[dict[str, object]]:
    """Return JSON-serializable release metadata in fixed component order."""

    return [
        {
            "name": spec.name,
            "mode": spec.mode,
            "supports_instance_count": spec.supports_instance_count,
            "supports_density": spec.supports_density,
            "supports_area": spec.supports_area,
            "supports_focus_density": spec.supports_focus_density,
        }
        for spec in spatial_component_specs(component_names)
    ]
