"""CLI entry point for Ontology Engine."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ontology_engine.core.config import OntologyConfig

console = Console()


@click.group()
@click.option("--config", "-c", default=None, help="Config file path (JSON or TOML)")
@click.pass_context
def main(ctx: click.Context, config: str | None) -> None:
    """Ontology Engine — Meeting transcripts → Structured knowledge graph."""
    import os

    from dotenv import load_dotenv

    load_dotenv()  # Load .env if present

    ctx.ensure_object(dict)
    if config:
        cfg = OntologyConfig.from_file(config)
    else:
        cfg = OntologyConfig()

    # Auto-populate API key from environment if not set in config
    if not cfg.llm.api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            cfg.llm.api_key = api_key

    ctx.obj["config"] = cfg


@main.command()
@click.option("--db-url", required=True, help="PostgreSQL connection URL")
@click.pass_context
def init(ctx: click.Context, db_url: str) -> None:
    """Initialize the database schema."""
    from ontology_engine.storage.schema import initialize_schema

    async def _init() -> None:
        await initialize_schema(db_url)
        console.print("[green]✓[/green] Schema initialized successfully")

    asyncio.run(_init())


@main.command()
@click.argument("file_path")
@click.option("--date", "-d", default=None, help="Meeting date (YYYY-MM-DD)")
@click.option("--participants", "-p", default=None, help="Comma-separated participant names")
@click.option("--db-url", default=None, help="PostgreSQL URL (optional, for storage)")
@click.option("--output", "-o", default=None, help="Output JSON file path")
@click.pass_context
def ingest(
    ctx: click.Context,
    file_path: str,
    date: str | None,
    participants: str | None,
    db_url: str | None,
    output: str | None,
) -> None:
    """Ingest a meeting transcript and extract structured knowledge."""
    from datetime import date as date_type

    from ontology_engine.pipeline.engine import PipelineEngine

    config: OntologyConfig = ctx.obj["config"]

    meeting_date = None
    if date:
        meeting_date = date_type.fromisoformat(date)

    participant_list = None
    if participants:
        participant_list = [p.strip() for p in participants.split(",")]

    async def _ingest() -> None:
        engine = await PipelineEngine.create(config, db_url=db_url)
        try:
            result = await engine.ingest(
                file_path,
                meeting_date=meeting_date,
                participants=participant_list,
            )

            if result.success:
                summary = result.summary()
                console.print(f"\n[green]✓[/green] Processed: {result.file}")
                console.print(f"  Time: {summary['time_ms']}ms")
                console.print(f"  Entities: {summary.get('entities', 0)}")
                console.print(f"  Links: {summary.get('links', 0)}")
                console.print(f"  Decisions: {summary.get('decisions', 0)}")
                console.print(f"  Action Items: {summary.get('action_items', 0)}")

                if result.validation:
                    console.print(f"  Auto-fixes: {result.validation.auto_fixes_applied}")
                    if result.validation.warnings:
                        console.print(
                            f"  [yellow]Warnings: {len(result.validation.warnings)}[/yellow]"
                        )

                if output and result.extraction:
                    out_data = result.extraction.model_dump(mode="json")
                    Path(output).write_text(
                        json.dumps(out_data, ensure_ascii=False, indent=2)
                    )
                    console.print(f"  Output: {output}")

            else:
                console.print(f"[red]✗[/red] Failed: {result.file}")
                console.print(f"  Error: {result.error}")
        finally:
            await engine.close()

    asyncio.run(_ingest())


@main.command()
@click.argument("directory")
@click.option("--pattern", default="*.md", help="File glob pattern")
@click.option("--db-url", default=None, help="PostgreSQL URL")
@click.pass_context
def ingest_dir(
    ctx: click.Context, directory: str, pattern: str, db_url: str | None
) -> None:
    """Ingest all meeting files in a directory."""
    from ontology_engine.pipeline.engine import PipelineEngine

    config: OntologyConfig = ctx.obj["config"]

    async def _run() -> None:
        engine = await PipelineEngine.create(config, db_url=db_url)
        try:
            results = await engine.ingest_directory(directory, pattern)

            table = Table(title="Ingestion Results")
            table.add_column("File", style="cyan")
            table.add_column("Status")
            table.add_column("Entities", justify="right")
            table.add_column("Decisions", justify="right")
            table.add_column("Actions", justify="right")
            table.add_column("Time (ms)", justify="right")

            for r in results:
                s = r.summary()
                status = "[green]✓[/green]" if r.success else f"[red]✗ {r.error}[/red]"
                table.add_row(
                    Path(r.file).name,
                    status,
                    str(s.get("entities", "-")),
                    str(s.get("decisions", "-")),
                    str(s.get("action_items", "-")),
                    str(s.get("time_ms", "-")),
                )

            console.print(table)
            console.print(
                f"\nTotal: {len(results)} files, "
                f"{sum(1 for r in results if r.success)} succeeded"
            )
        finally:
            await engine.close()

    asyncio.run(_run())


@main.command()
@click.option("--db-url", required=True, help="PostgreSQL URL")
@click.argument("query_text")
@click.pass_context
def query(ctx: click.Context, db_url: str, query_text: str) -> None:
    """Query the ontology (simple keyword search)."""
    from ontology_engine.storage.repository import OntologyRepository

    async def _query() -> None:
        repo = await OntologyRepository.create(db_url)
        try:
            # Try decision search first
            decisions = await repo.query_decisions(keyword=query_text)
            if decisions:
                console.print(f"\n[bold]Decisions matching '{query_text}':[/bold]")
                for d in decisions:
                    console.print(f"  • {d.name}")
                    if d.properties.get("detail"):
                        console.print(f"    {d.properties['detail'][:100]}")

            # Entity search
            entities = await repo.find_entity_by_name(query_text)
            if entities:
                console.print(f"\n[bold]Entities matching '{query_text}':[/bold]")
                for e in entities:
                    console.print(f"  • [{e.entity_type.value}] {e.name}")

            if not decisions and not entities:
                console.print(f"No results for '{query_text}'")
        finally:
            await repo.close()

    asyncio.run(_query())


@main.command()
@click.option("--db-url", required=True, help="PostgreSQL URL")
@click.pass_context
def stats(ctx: click.Context, db_url: str) -> None:
    """Show ontology statistics."""
    from ontology_engine.storage.repository import OntologyRepository

    async def _stats() -> None:
        repo = await OntologyRepository.create(db_url)
        try:
            s = await repo.stats()
            table = Table(title="Ontology Stats")
            table.add_column("Metric")
            table.add_column("Count", justify="right")
            for k, v in s.items():
                table.add_row(k, str(v))
            console.print(table)
        finally:
            await repo.close()

    asyncio.run(_stats())


# =========================================================================
# Gold Layer Commands
# =========================================================================


@main.group()
@click.pass_context
def gold(ctx: click.Context) -> None:
    """Gold layer commands — entity resolution, search, and graph queries."""
    pass


@gold.command("build")
@click.option("--db-url", required=True, help="PostgreSQL connection URL")
@click.option("--full", is_flag=True, default=False, help="Full rebuild (clear Gold first)")
@click.pass_context
def gold_build(ctx: click.Context, db_url: str, full: bool) -> None:
    """Build or incrementally update the Gold layer from Silver."""
    from ontology_engine.fusion.gold_builder import GoldBuilder

    config: OntologyConfig = ctx.obj["config"]

    async def _build() -> None:
        builder = await GoldBuilder.create(db_url, config)
        try:
            result = await builder.build_gold(full=full)
            s = result.summary()

            table = Table(title="Gold Build Result")
            table.add_column("Metric")
            table.add_column("Count", justify="right")
            for k, v in s.items():
                table.add_row(k.replace("_", " ").title(), str(v))
            console.print(table)

            if result.review_candidates:
                console.print(
                    f"\n[yellow]⚠ {len(result.review_candidates)} entity pairs "
                    f"need manual review:[/yellow]"
                )
                for mc in result.review_candidates:
                    console.print(
                        f"  • '{mc.entity_a.name}' ↔ '{mc.entity_b.name}' "
                        f"(sim={mc.similarity:.3f}, reason={mc.match_reason})"
                    )

            if result.errors:
                for err in result.errors:
                    console.print(f"  [red]Error: {err}[/red]")
        finally:
            await builder.close()

    asyncio.run(_build())


@gold.command("query")
@click.option("--db-url", required=True, help="PostgreSQL connection URL")
@click.option("--type", "entity_type", default=None, help="Entity type filter")
@click.option("--status", default="active", help="Status filter")
@click.option("--limit", default=20, help="Max results")
@click.pass_context
def gold_query(
    ctx: click.Context,
    db_url: str,
    entity_type: str | None,
    status: str,
    limit: int,
) -> None:
    """Query Gold entities by type and status."""
    from ontology_engine.storage.gold_repository import GoldRepository

    async def _query() -> None:
        repo = await GoldRepository.create(db_url)
        try:
            entities = await repo.query(
                entity_type=entity_type, status=status, limit=limit
            )
            if not entities:
                console.print("No Gold entities found.")
                return

            table = Table(title=f"Gold Entities ({len(entities)})")
            table.add_column("ID", style="dim")
            table.add_column("Type")
            table.add_column("Name", style="cyan")
            table.add_column("Aliases")
            table.add_column("Sources", justify="right")
            table.add_column("Confidence", justify="right")

            for e in entities:
                table.add_row(
                    e.id[:12] + "…" if len(e.id) > 12 else e.id,
                    e.entity_type,
                    e.canonical_name,
                    ", ".join(e.aliases[:3]) + ("…" if len(e.aliases) > 3 else ""),
                    str(e.source_count),
                    f"{e.confidence:.2f}",
                )

            console.print(table)
        finally:
            await repo.close()

    asyncio.run(_query())


@gold.command("search")
@click.argument("text")
@click.option("--db-url", required=True, help="PostgreSQL connection URL")
@click.option("--limit", default=10, help="Max results")
@click.pass_context
def gold_search(ctx: click.Context, text: str, db_url: str, limit: int) -> None:
    """Search Gold entities by semantic similarity."""
    from ontology_engine.storage.gold_repository import GoldRepository

    async def _search() -> None:
        repo = await GoldRepository.create(db_url)
        try:
            entities = await repo.search(text=text, limit=limit)
            if not entities:
                console.print(f"No results for '{text}'")
                return

            table = Table(title=f"Search Results for '{text}'")
            table.add_column("Type")
            table.add_column("Name", style="cyan")
            table.add_column("Similarity", justify="right")
            table.add_column("Aliases")
            table.add_column("Sources", justify="right")

            for e in entities:
                sim_str = f"{e.similarity:.3f}" if e.similarity is not None else "-"
                table.add_row(
                    e.entity_type,
                    e.canonical_name,
                    sim_str,
                    ", ".join(e.aliases[:3]),
                    str(e.source_count),
                )

            console.print(table)
        finally:
            await repo.close()

    asyncio.run(_search())


@gold.command("stats")
@click.option("--db-url", required=True, help="PostgreSQL connection URL")
@click.pass_context
def gold_stats(ctx: click.Context, db_url: str) -> None:
    """Show Gold layer statistics."""
    from ontology_engine.storage.gold_repository import GoldRepository

    async def _stats() -> None:
        repo = await GoldRepository.create(db_url)
        try:
            s = await repo.stats()

            table = Table(title="Gold Layer Stats")
            table.add_column("Metric")
            table.add_column("Value", justify="right")

            table.add_row("Total Entities", str(s.total_entities))
            table.add_row("Total Links", str(s.total_links))
            table.add_row("Avg Sources/Entity", f"{s.avg_source_count:.1f}")
            table.add_row("With Embeddings", str(s.entities_with_embeddings))

            if s.entities_by_type:
                table.add_row("", "")
                table.add_row("[bold]Entities by Type[/bold]", "")
                for t, c in s.entities_by_type.items():
                    table.add_row(f"  {t}", str(c))

            if s.links_by_type:
                table.add_row("", "")
                table.add_row("[bold]Links by Type[/bold]", "")
                for t, c in s.links_by_type.items():
                    table.add_row(f"  {t}", str(c))

            console.print(table)
        finally:
            await repo.close()

    asyncio.run(_stats())


if __name__ == "__main__":
    main()
