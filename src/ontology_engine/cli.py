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


if __name__ == "__main__":
    main()
