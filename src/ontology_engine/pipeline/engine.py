"""Pipeline Engine — orchestrates the full Meeting-to-Ontology flow.

Usage:
    engine = await PipelineEngine.create(config)
    result = await engine.ingest("path/to/meeting.md")
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from ontology_engine.core.config import OntologyConfig
from ontology_engine.core.errors import ExtractionError
from ontology_engine.core.types import (
    Entity,
    EntityType,
    ExtractionResult,
    Link,
    LinkType,
    Provenance,
    ValidationResult,
)
from ontology_engine.llm.base import LLMProvider
from ontology_engine.pipeline.extractor import StructuredExtractor
from ontology_engine.pipeline.preprocessor import MeetingPreprocessor, ProcessedMeeting
from ontology_engine.pipeline.validator import ExtractionValidator
from ontology_engine.storage.bronze import BronzeRepository
from ontology_engine.storage.repository import OntologyRepository


class PipelineEngine:
    """End-to-end pipeline: file → bronze → preprocess → extract → validate → store."""

    def __init__(
        self,
        llm: LLMProvider,
        repo: OntologyRepository | None,
        config: OntologyConfig,
        bronze: BronzeRepository | None = None,
    ):
        self.llm = llm
        self.repo = repo
        self.bronze = bronze
        self.config = config
        self.preprocessor = MeetingPreprocessor(llm, config)
        self.extractor = StructuredExtractor(llm, config)
        self.validator = ExtractionValidator(config)

    @classmethod
    async def create(
        cls,
        config: OntologyConfig,
        db_url: str | None = None,
    ) -> PipelineEngine:
        """Factory: create engine with LLM provider and optional DB."""
        llm = _create_llm(config)

        repo = None
        bronze = None
        url = db_url or config.database.url
        if url:
            repo = await OntologyRepository.create(url, config.database.db_schema)
            bronze = BronzeRepository(repo._pool, config.database.db_schema)

        return cls(llm, repo, config, bronze=bronze)

    async def ingest(
        self,
        file_path: str,
        *,
        meeting_date: date | None = None,
        participants: list[str] | None = None,
        source_type: str = "meeting_transcript",
        content_format: str = "text",
    ) -> IngestResult:
        """Ingest a single meeting transcript file.

        Flow: file → Bronze (dedup) → preprocess → extract → validate → store.
        If the document already exists in Bronze (same hash), returns early
        with the existing doc_id and skips extraction.

        Returns a structured result with extraction data, validation report,
        and storage IDs (if DB is connected).
        """
        t0 = time.monotonic()

        # Read file
        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")
        raw_text = path.read_text(encoding="utf-8")

        # Detect meeting date from filename if not provided
        if meeting_date is None:
            meeting_date = self._parse_date_from_filename(path.name)

        # Stage 0: Bronze Layer — immutable raw document storage (dedup)
        bronze_doc_id: str | None = None
        if self.bronze:
            bronze_doc_id, is_new = await self.bronze.ingest(
                content=raw_text,
                source_type=source_type,
                source_uri=str(path),
                content_format=content_format,
                metadata={
                    "meeting_date": meeting_date.isoformat() if meeting_date else None,
                    "participants": participants or [],
                },
                ingested_by="pipeline",
            )
            if not is_new:
                # Document already processed — skip extraction
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                return IngestResult(
                    file=str(path),
                    meeting_date=meeting_date,
                    bronze_doc_id=bronze_doc_id,
                    skipped_duplicate=True,
                    processing_time_ms=elapsed_ms,
                )

        # Stage 1: Preprocessing
        processed = await self.preprocessor.process(
            raw_text,
            meeting_date=meeting_date,
            participants=participants,
        )
        processed.metadata["source_file"] = str(path)

        # Stage 2: Extraction (4-pass)
        extraction = await self.extractor.extract(processed)
        extraction.source_file = str(path)

        # Stage 3: Validation
        validation = self.validator.validate(extraction)

        # Stage 4: Store (if DB connected and validation passed)
        stored_ids: dict[str, list[str]] = {
            "entities": [],
            "links": [],
            "provenance": [],
        }
        if self.repo and validation.is_valid and validation.extraction:
            stored_ids = await self._store_results(
                validation.extraction, str(path), bronze_doc_id=bronze_doc_id
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return IngestResult(
            file=str(path),
            meeting_date=meeting_date,
            participants=processed.participants,
            extraction=validation.extraction or extraction,
            validation=validation,
            stored_ids=stored_ids,
            bronze_doc_id=bronze_doc_id,
            processing_time_ms=elapsed_ms,
        )

    async def ingest_directory(
        self, directory: str, pattern: str = "*.md"
    ) -> list[IngestResult]:
        """Ingest all matching files in a directory."""
        dir_path = Path(directory)
        results: list[IngestResult] = []
        for f in sorted(dir_path.glob(pattern)):
            try:
                result = await self.ingest(str(f))
                results.append(result)
            except Exception as e:
                results.append(
                    IngestResult(
                        file=str(f),
                        error=str(e),
                    )
                )
        return results

    # =========================================================================
    # Storage
    # =========================================================================

    async def _store_results(
        self,
        extraction: ExtractionResult,
        source_file: str,
        *,
        bronze_doc_id: str | None = None,
    ) -> dict[str, list[str]]:
        """Persist extraction results to PostgreSQL."""
        assert self.repo is not None

        ids: dict[str, list[str]] = {"entities": [], "links": [], "provenance": []}
        entity_name_to_id: dict[str, str] = {}

        # Store entities
        for ext_ent in extraction.entities:
            entity = Entity(
                entity_type=ext_ent.entity_type,
                name=ext_ent.name,
                properties=ext_ent.properties,
                aliases=ext_ent.aliases,
                confidence=ext_ent.confidence,
                created_by="pipeline",
            )

            # Check for existing entity (upsert logic)
            existing = await self.repo.find_entity_by_name(ext_ent.name, ext_ent.entity_type)
            if existing:
                # Update existing
                ent_id = existing[0].id
                assert ent_id is not None
                await self.repo.update_entity(
                    ent_id,
                    {
                        "properties": {**existing[0].properties, **ext_ent.properties},
                        "aliases": list(set(existing[0].aliases + ext_ent.aliases)),
                        "confidence": max(existing[0].confidence, ext_ent.confidence),
                    },
                    updated_by="pipeline",
                )
            else:
                ent_id = await self.repo.create_entity(entity)

            entity_name_to_id[ext_ent.name.lower()] = ent_id
            ids["entities"].append(ent_id)

            # Provenance
            prov_id = await self.repo.create_provenance(
                Provenance(
                    entity_id=ent_id,
                    source_document_id=bronze_doc_id,
                    source_type="meeting_transcript",
                    source_file=source_file,
                    source_meeting_date=extraction.meeting_date,
                    source_participants=extraction.participants,
                    source_segment=ext_ent.context,
                    extraction_model=extraction.extraction_model,
                    extraction_pass="pass1",
                    created_by="pipeline",
                )
            )
            ids["provenance"].append(prov_id)

        # Store decisions as entities
        for dec in extraction.decisions:
            dec_entity = Entity(
                entity_type=EntityType.DECISION,
                name=dec.summary,
                properties={
                    "detail": dec.detail,
                    "decision_type": dec.decision_type.value,
                    "rationale": dec.rationale,
                    "conditions": dec.conditions,
                    "status": "active",
                },
                confidence=dec.confidence,
                created_by="pipeline",
            )
            dec_id = await self.repo.create_entity(dec_entity)
            entity_name_to_id[dec.summary.lower()] = dec_id
            ids["entities"].append(dec_id)

            # Link: Person → makes → Decision
            if dec.made_by:
                maker_id = entity_name_to_id.get(dec.made_by.lower())
                if maker_id:
                    link = Link(
                        link_type=LinkType.MAKES,
                        source_entity_id=maker_id,
                        target_entity_id=dec_id,
                        created_by="pipeline",
                    )
                    link_id = await self.repo.create_link(link)
                    ids["links"].append(link_id)

        # Store action items as entities
        for act in extraction.action_items:
            act_entity = Entity(
                entity_type=EntityType.ACTION_ITEM,
                name=act.task,
                properties={
                    "priority": act.priority,
                    "status": "pending",
                    "due_date": act.due_date.isoformat() if act.due_date else None,
                    "completion_criteria": act.completion_criteria,
                },
                confidence=act.confidence,
                created_by="pipeline",
            )
            act_id = await self.repo.create_entity(act_entity)
            ids["entities"].append(act_id)

            # Link: ActionItem → assigned_to → Person
            if act.owner:
                owner_id = entity_name_to_id.get(act.owner.lower())
                if owner_id:
                    link = Link(
                        link_type=LinkType.ASSIGNED_TO,
                        source_entity_id=act_id,
                        target_entity_id=owner_id,
                        created_by="pipeline",
                    )
                    link_id = await self.repo.create_link(link)
                    ids["links"].append(link_id)

        # Store extracted links
        for ext_link in extraction.links:
            src_id = entity_name_to_id.get(ext_link.source_name.lower())
            tgt_id = entity_name_to_id.get(ext_link.target_name.lower())
            if src_id and tgt_id:
                link = Link(
                    link_type=ext_link.link_type,
                    source_entity_id=src_id,
                    target_entity_id=tgt_id,
                    properties=ext_link.properties,
                    confidence=ext_link.confidence,
                    created_by="pipeline",
                )
                try:
                    link_id = await self.repo.create_link(link)
                    ids["links"].append(link_id)
                except Exception:
                    pass  # Duplicate link constraint

        return ids

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _parse_date_from_filename(filename: str) -> date | None:
        """Try to extract a date from filename like '20260208_meeting.md'."""
        import re

        m = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

        return None

    async def close(self) -> None:
        """Clean up resources."""
        if self.repo:
            await self.repo.close()
        # Note: bronze shares the repo pool, so no separate close needed
        # unless it was created independently
        if self.bronze and not self.repo:
            await self.bronze.close()


class IngestResult:
    """Result of processing a single meeting file."""

    def __init__(
        self,
        file: str = "",
        meeting_date: date | None = None,
        participants: list[str] | None = None,
        extraction: ExtractionResult | None = None,
        validation: ValidationResult | None = None,
        stored_ids: dict[str, list[str]] | None = None,
        processing_time_ms: int = 0,
        error: str | None = None,
        bronze_doc_id: str | None = None,
        skipped_duplicate: bool = False,
    ):
        self.file = file
        self.meeting_date = meeting_date
        self.participants = participants or []
        self.extraction = extraction
        self.validation = validation
        self.stored_ids = stored_ids or {}
        self.processing_time_ms = processing_time_ms
        self.error = error
        self.bronze_doc_id = bronze_doc_id
        self.skipped_duplicate = skipped_duplicate

    @property
    def success(self) -> bool:
        return self.error is None

    def summary(self) -> dict[str, Any]:
        """Quick summary for logging."""
        s: dict[str, Any] = {
            "file": self.file,
            "success": self.success,
            "time_ms": self.processing_time_ms,
        }
        if self.bronze_doc_id:
            s["bronze_doc_id"] = self.bronze_doc_id
        if self.skipped_duplicate:
            s["skipped_duplicate"] = True
        if self.error:
            s["error"] = self.error
        if self.extraction:
            s["entities"] = len(self.extraction.entities)
            s["links"] = len(self.extraction.links)
            s["decisions"] = len(self.extraction.decisions)
            s["action_items"] = len(self.extraction.action_items)
        if self.validation:
            s["valid"] = self.validation.is_valid
            s["auto_fixes"] = self.validation.auto_fixes_applied
            s["errors"] = len(self.validation.errors)
            s["warnings"] = len(self.validation.warnings)
        if self.stored_ids:
            s["stored"] = {k: len(v) for k, v in self.stored_ids.items()}
        return s


def _create_llm(config: OntologyConfig) -> LLMProvider:
    """Create an LLM provider based on config."""
    provider = config.llm.provider.lower()
    if provider == "gemini":
        from ontology_engine.llm.gemini import GeminiProvider

        return GeminiProvider(config.llm)
    elif provider == "openai":
        from ontology_engine.llm.openai import OpenAIProvider

        return OpenAIProvider(config.llm)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
