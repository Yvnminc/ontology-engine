"""Stage 3: 3-Layer auto-correction and validation.

Layer 1: Factual correction (name matching, date logic, known entities)
Layer 2: Semantic correction (contradiction detection) — Phase 2, disabled
Layer 3: Consistency checks (orphan entities, broken refs, schema compliance)
"""

from __future__ import annotations

from ontology_engine.core.config import OntologyConfig
from ontology_engine.core.types import (
    EntityType,
    ExtractionResult,
    ExtractedEntity,
    ValidationError as VError,
    ValidationResult,
)


class ExtractionValidator:
    """3-layer validation pipeline for extraction results."""

    def __init__(self, config: OntologyConfig, known_entities: dict[str, list[str]] | None = None):
        self.config = config
        # Build reverse alias map: alias → canonical name
        self._alias_map: dict[str, str] = {}
        aliases = known_entities or config.known_entities.aliases
        for canonical, alias_list in aliases.items():
            self._alias_map[canonical.lower()] = canonical
            for alias in alias_list:
                self._alias_map[alias.lower()] = canonical

    def validate(self, result: ExtractionResult) -> ValidationResult:
        """Run all enabled validation layers."""
        errors: list[VError] = []
        warnings: list[VError] = []
        fixes = 0

        # Layer 1: Factual correction
        if self.config.pipeline.enable_factual_correction:
            layer1_errors, layer1_fixes = self._layer1_factual(result)
            for e in layer1_errors:
                (errors if e.severity == "error" else warnings).append(e)
            fixes += layer1_fixes

        # Layer 2: Semantic (disabled for Phase 1)
        # if self.config.pipeline.enable_semantic_correction:
        #     ...

        # Layer 3: Consistency
        if self.config.pipeline.enable_consistency_check:
            layer3_errors = self._layer3_consistency(result)
            for e in layer3_errors:
                (errors if e.severity == "error" else warnings).append(e)

        is_valid = not any(e.severity == "error" for e in errors)

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            auto_fixes_applied=fixes,
            extraction=result,
        )

    # =========================================================================
    # Layer 1: Factual Correction
    # =========================================================================

    def _layer1_factual(self, result: ExtractionResult) -> tuple[list[VError], int]:
        """Check and correct factual issues: name matching, type validation."""
        errors: list[VError] = []
        fixes = 0

        for entity in result.entities:
            # 1a. Name normalization via alias map
            canonical = self._resolve_name(entity.name)
            if canonical and canonical != entity.name:
                old_name = entity.name
                entity.name = canonical
                if old_name not in entity.aliases:
                    entity.aliases.append(old_name)
                fixes += 1
                errors.append(
                    VError(
                        layer="factual",
                        severity="info",
                        entity_name=canonical,
                        field="name",
                        message=f"Name normalized: '{old_name}' → '{canonical}'",
                        auto_fixed=True,
                    )
                )

            # 1b. Check aliases too
            for alias in entity.aliases:
                canonical_alias = self._resolve_name(alias)
                if canonical_alias and canonical_alias != entity.name:
                    errors.append(
                        VError(
                            layer="factual",
                            severity="warning",
                            entity_name=entity.name,
                            field="aliases",
                            message=f"Alias '{alias}' maps to different entity '{canonical_alias}'",
                            suggestion=f"Consider merging with '{canonical_alias}'",
                        )
                    )

            # 1c. Empty required fields
            if entity.entity_type == EntityType.PERSON:
                if not entity.properties.get("role"):
                    errors.append(
                        VError(
                            layer="factual",
                            severity="warning",
                            entity_name=entity.name,
                            field="properties.role",
                            message="Person entity missing 'role' field",
                        )
                    )

            # 1d. Confidence threshold
            if entity.confidence < self.config.pipeline.min_confidence:
                errors.append(
                    VError(
                        layer="factual",
                        severity="warning",
                        entity_name=entity.name,
                        field="confidence",
                        message=f"Low confidence: {entity.confidence:.2f}",
                        suggestion="Review extraction or raise threshold",
                    )
                )

        # Check decisions
        for dec in result.decisions:
            canonical_maker = self._resolve_name(dec.made_by)
            if canonical_maker and canonical_maker != dec.made_by:
                dec.made_by = canonical_maker
                fixes += 1

            for i, p in enumerate(dec.participants):
                canonical_p = self._resolve_name(p)
                if canonical_p and canonical_p != p:
                    dec.participants[i] = canonical_p
                    fixes += 1

        # Check action items
        for act in result.action_items:
            canonical_owner = self._resolve_name(act.owner)
            if canonical_owner and canonical_owner != act.owner:
                act.owner = canonical_owner
                fixes += 1

            for i, a in enumerate(act.assignees):
                canonical_a = self._resolve_name(a)
                if canonical_a and canonical_a != a:
                    act.assignees[i] = canonical_a
                    fixes += 1

        return errors, fixes

    # =========================================================================
    # Layer 3: Consistency Checks
    # =========================================================================

    def _layer3_consistency(self, result: ExtractionResult) -> list[VError]:
        """Check structural consistency of extraction results."""
        errors: list[VError] = []
        entity_names = {e.name.lower() for e in result.entities}

        # 3a. Links reference existing entities
        for link in result.links:
            if link.source_name.lower() not in entity_names:
                errors.append(
                    VError(
                        layer="consistency",
                        severity="warning",
                        entity_name=link.source_name,
                        field="source_name",
                        message=f"Link source '{link.source_name}' not in extracted entities",
                        suggestion="Entity may have been filtered by confidence threshold",
                    )
                )
            if link.target_name.lower() not in entity_names:
                errors.append(
                    VError(
                        layer="consistency",
                        severity="warning",
                        entity_name=link.target_name,
                        field="target_name",
                        message=f"Link target '{link.target_name}' not in extracted entities",
                    )
                )

        # 3b. Action items reference known people
        for act in result.action_items:
            if act.owner and act.owner.lower() not in entity_names:
                errors.append(
                    VError(
                        layer="consistency",
                        severity="warning",
                        entity_name=act.owner,
                        field="owner",
                        message=f"Action item owner '{act.owner}' not in extracted entities",
                    )
                )

        # 3c. Decision makers are known
        for dec in result.decisions:
            if dec.made_by and dec.made_by.lower() not in entity_names:
                errors.append(
                    VError(
                        layer="consistency",
                        severity="warning",
                        entity_name=dec.made_by,
                        field="made_by",
                        message=f"Decision maker '{dec.made_by}' not in extracted entities",
                    )
                )

        # 3d. Duplicate entity names
        seen_names: dict[str, int] = {}
        for ent in result.entities:
            key = ent.name.lower()
            seen_names[key] = seen_names.get(key, 0) + 1
        for name, count in seen_names.items():
            if count > 1:
                errors.append(
                    VError(
                        layer="consistency",
                        severity="error",
                        entity_name=name,
                        message=f"Duplicate entity name: '{name}' appears {count} times",
                        suggestion="Merge duplicates",
                    )
                )

        return errors

    # =========================================================================
    # Helpers
    # =========================================================================

    def _resolve_name(self, name: str) -> str | None:
        """Resolve a name or alias to its canonical form."""
        if not name:
            return None
        return self._alias_map.get(name.lower().strip())
