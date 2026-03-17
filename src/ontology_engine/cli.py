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
@click.option("--schema", "-s", "schema_name", default=None, help="Domain schema name (e.g. edtech, finance)")
@click.pass_context
def ingest(
    ctx: click.Context,
    file_path: str,
    date: str | None,
    participants: str | None,
    db_url: str | None,
    output: str | None,
    schema_name: str | None,
) -> None:
    """Ingest a meeting transcript and extract structured knowledge."""
    from datetime import date as date_type

    from ontology_engine.core.schema_registry import DomainSchema, SchemaRegistry
    from ontology_engine.pipeline.engine import PipelineEngine

    config: OntologyConfig = ctx.obj["config"]

    meeting_date = None
    if date:
        meeting_date = date_type.fromisoformat(date)

    participant_list = None
    if participants:
        participant_list = [p.strip() for p in participants.split(",")]

    # Load domain schema if specified
    domain_schema = None
    if schema_name:
        domain_schema = _load_schema_by_name(schema_name)

    async def _ingest() -> None:
        engine = await PipelineEngine.create(
            config, db_url=db_url, domain_schema=domain_schema,
        )

        try:
            result = await engine.ingest(
                file_path,
                meeting_date=meeting_date,
                participants=participant_list,
            )

            if result.success:
                summary = result.summary()
                console.print(f"\n[green]✓[/green] Processed: {result.file}")
                if domain_schema:
                    console.print(f"  Schema: {domain_schema.domain} v{domain_schema.version}")
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
@click.option("--schema", "-s", "schema_name", default=None, help="Domain schema name (e.g. edtech, finance)")
@click.pass_context
def ingest_dir(
    ctx: click.Context, directory: str, pattern: str, db_url: str | None,
    schema_name: str | None,
) -> None:
    """Ingest all meeting files in a directory."""
    from ontology_engine.pipeline.engine import PipelineEngine

    config: OntologyConfig = ctx.obj["config"]

    domain_schema = None
    if schema_name:
        domain_schema = _load_schema_by_name(schema_name)

    async def _run() -> None:
        engine = await PipelineEngine.create(
            config, db_url=db_url, domain_schema=domain_schema,
        )
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


@main.group()
def bronze() -> None:
    """Manage Bronze layer (raw document store)."""
    pass


@bronze.command("list")
@click.option("--db-url", required=True, help="PostgreSQL URL")
@click.option("--source-type", "-t", default=None, help="Filter by source type")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--offset", default=0, help="Offset for pagination")
@click.pass_context
def bronze_list(
    ctx: click.Context,
    db_url: str,
    source_type: str | None,
    limit: int,
    offset: int,
) -> None:
    """List Bronze documents."""
    from ontology_engine.storage.bronze import BronzeRepository

    async def _run() -> None:
        repo = await BronzeRepository.create(db_url)
        try:
            docs = await repo.list(source_type=source_type, limit=limit, offset=offset)
            if not docs:
                console.print("[dim]No bronze documents found.[/dim]")
                return

            table = Table(title="Bronze Documents")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Source Type")
            table.add_column("Source URI", max_width=40)
            table.add_column("Format")
            table.add_column("Size", justify="right")
            table.add_column("Ingested At")
            table.add_column("Ingested By")

            for doc in docs:
                table.add_row(
                    doc.id,
                    doc.source_type,
                    doc.source_uri or "-",
                    doc.content_format,
                    str(len(doc.content)),
                    doc.ingested_at.strftime("%Y-%m-%d %H:%M") if doc.ingested_at else "-",
                    doc.ingested_by,
                )

            console.print(table)
            console.print(f"\n[dim]Showing {len(docs)} documents[/dim]")
        finally:
            await repo.close()

    asyncio.run(_run())


@bronze.command("show")
@click.argument("doc_id")
@click.option("--db-url", required=True, help="PostgreSQL URL")
@click.option("--content/--no-content", default=True, help="Show document content")
@click.pass_context
def bronze_show(ctx: click.Context, doc_id: str, db_url: str, content: bool) -> None:
    """Show a single Bronze document."""
    from ontology_engine.storage.bronze import BronzeRepository

    async def _run() -> None:
        repo = await BronzeRepository.create(db_url)
        try:
            doc = await repo.get(doc_id)
            if doc is None:
                console.print(f"[red]Document not found: {doc_id}[/red]")
                sys.exit(1)

            console.print(f"\n[bold]Document:[/bold] {doc.id}")
            console.print(f"  Source Type:   {doc.source_type}")
            console.print(f"  Source URI:    {doc.source_uri or '-'}")
            console.print(f"  Source Hash:   {doc.source_hash}")
            console.print(f"  Format:        {doc.content_format}")
            console.print(f"  Language:      {doc.language}")
            console.print(f"  Size:          {len(doc.content)} chars")
            console.print(f"  Ingested At:   {doc.ingested_at}")
            console.print(f"  Ingested By:   {doc.ingested_by}")

            if doc.metadata:
                console.print(f"  Metadata:      {json.dumps(doc.metadata, ensure_ascii=False)}")

            if content:
                console.print("\n[bold]Content:[/bold]")
                console.print(doc.content[:2000])
                if len(doc.content) > 2000:
                    console.print(f"\n[dim]... truncated ({len(doc.content)} total chars)[/dim]")
        finally:
            await repo.close()

    asyncio.run(_run())


# =============================================================================
# Schema commands
# =============================================================================

_SCHEMA_DIR = Path(__file__).parent.parent.parent.parent / "domain_schemas"


def _find_schema_dir() -> Path:
    """Find the domain_schemas directory."""
    # Try relative to package
    if _SCHEMA_DIR.exists():
        return _SCHEMA_DIR
    # Try CWD
    cwd_dir = Path.cwd() / "domain_schemas"
    if cwd_dir.exists():
        return cwd_dir
    return _SCHEMA_DIR  # Return default even if missing


def _load_schema_by_name(name: str) -> "DomainSchema":
    """Load a schema by domain name from the schema directory."""
    from ontology_engine.core.schema_registry import DomainSchema

    schema_dir = _find_schema_dir()
    # Try exact name.yaml
    for ext in (".yaml", ".yml"):
        path = schema_dir / f"{name}{ext}"
        if path.exists():
            return DomainSchema.from_yaml(path)
    raise click.ClickException(f"Schema '{name}' not found in {schema_dir}")


@main.group()
def schema() -> None:
    """Manage domain schemas."""
    pass


@schema.command("list")
def schema_list() -> None:
    """List available domain schemas."""
    from ontology_engine.core.schema_registry import DomainSchema

    schema_dir = _find_schema_dir()
    if not schema_dir.exists():
        console.print(f"[yellow]Schema directory not found: {schema_dir}[/yellow]")
        return

    yaml_files = sorted(schema_dir.glob("*.yaml")) + sorted(schema_dir.glob("*.yml"))
    if not yaml_files:
        console.print("[dim]No schemas found.[/dim]")
        return

    for f in yaml_files:
        try:
            s = DomainSchema.from_yaml(f)
            console.print(
                f"  {s.domain:<15s} v{s.version:<8s} "
                f"{len(s.entity_types)} entity types, {len(s.link_types)} link types"
            )
        except Exception as exc:
            console.print(f"  [red]✗ {f.name}: {exc}[/red]")


@schema.command("validate")
@click.argument("path")
def schema_validate(path: str) -> None:
    """Validate a domain schema YAML file."""
    from ontology_engine.core.schema_registry import DomainSchema

    try:
        s = DomainSchema.from_yaml(path)
        console.print(
            f"[green]✓[/green] Schema '{s.domain}' v{s.version} is valid: "
            f"{len(s.entity_types)} entity types, {len(s.link_types)} link types"
        )
    except Exception as exc:
        console.print(f"[red]✗[/red] Validation failed: {exc}")
        sys.exit(1)


@schema.command("show")
@click.argument("name_or_path")
def schema_show(name_or_path: str) -> None:
    """Show details of a domain schema."""
    from ontology_engine.core.schema_registry import DomainSchema

    # Try as file path first
    p = Path(name_or_path)
    if p.exists():
        s = DomainSchema.from_yaml(p)
    else:
        s = _load_schema_by_name(name_or_path)

    console.print(f"\n[bold]{s.domain}[/bold] v{s.version}")
    if s.description:
        console.print(f"  {s.description}")

    console.print(f"\n[bold]Entity Types ({len(s.entity_types)}):[/bold]")
    for et in s.entity_types:
        props = ", ".join(p.name for p in et.properties)
        console.print(f"  • {et.name}: {et.description or '-'}")
        if props:
            console.print(f"    Properties: {props}")

    console.print(f"\n[bold]Link Types ({len(s.link_types)}):[/bold]")
    for lt in s.link_types:
        src = ", ".join(lt.source_types) if lt.source_types else "*"
        tgt = ", ".join(lt.target_types) if lt.target_types else "*"
        console.print(f"  • {lt.name}: {src} → {tgt}")
        if lt.description:
            console.print(f"    {lt.description}")


# =============================================================================
# Kinetic Layer commands
# =============================================================================


@main.group()
def action() -> None:
    """Manage Kinetic Layer action types."""
    pass


@action.command("list")
@click.option("--schema-dir", default=None, help="Directory containing action YAML files")
def action_list(schema_dir: str | None) -> None:
    """List all registered action types."""
    from ontology_engine.kinetic.action_types import ActionRegistry, load_actions_from_yaml

    registry = ActionRegistry()
    _load_actions_into_registry(registry, schema_dir)

    actions = registry.list()
    if not actions:
        console.print("[dim]No action types found.[/dim]")
        return

    table = Table(title="Action Types")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Idempotent", justify="center")
    table.add_column("Reversible", justify="center")
    table.add_column("Preconditions", justify="right")

    for a in actions:
        table.add_row(
            a.name,
            a.description[:60] + ("..." if len(a.description) > 60 else ""),
            "✓" if a.idempotent else "✗",
            "✓" if a.reversible else "✗",
            str(len(a.preconditions)),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(actions)} action types[/dim]")


@action.command("show")
@click.argument("name")
@click.option("--schema-dir", default=None, help="Directory containing action YAML files")
def action_show(name: str, schema_dir: str | None) -> None:
    """Show details of a specific action type."""
    from ontology_engine.kinetic.action_types import ActionRegistry

    registry = ActionRegistry()
    _load_actions_into_registry(registry, schema_dir)

    try:
        a = registry.get(name)
    except KeyError:
        console.print(f"[red]Action type not found: {name}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]{a.name}[/bold]")
    console.print(f"  Description:  {a.description}")
    console.print(f"  Idempotent:   {'Yes' if a.idempotent else 'No'}")
    console.print(f"  Reversible:   {'Yes' if a.reversible else 'No'}")

    if a.input_schema:
        console.print(f"\n  [bold]Input Schema:[/bold]")
        console.print(f"    {json.dumps(a.input_schema, indent=2)}")
    if a.output_schema:
        console.print(f"\n  [bold]Output Schema:[/bold]")
        console.print(f"    {json.dumps(a.output_schema, indent=2)}")
    if a.preconditions:
        console.print(f"\n  [bold]Preconditions:[/bold]")
        for p in a.preconditions:
            console.print(f"    • {p}")
    if a.postconditions:
        console.print(f"\n  [bold]Postconditions:[/bold]")
        for p in a.postconditions:
            console.print(f"    • {p}")
    if a.side_effects:
        console.print(f"\n  [bold]Side Effects:[/bold]")
        for s in a.side_effects:
            console.print(f"    • {s}")


@action.command("execute")
@click.argument("name")
@click.option("--params", "-p", required=True, help="JSON string of action parameters")
@click.option("--actor", "-a", default="cli", help="Actor identity (default: cli)")
@click.option("--schema-dir", default=None, help="Directory containing action YAML files")
def action_execute(name: str, params: str, actor: str, schema_dir: str | None) -> None:
    """Execute an action type (dry-run: no DB handler by default)."""
    from ontology_engine.kinetic.action_executor import ActionExecutor, ExecutionContext
    from ontology_engine.kinetic.action_types import ActionRegistry
    from ontology_engine.kinetic.audit_trail import AuditTrail

    registry = ActionRegistry()
    _load_actions_into_registry(registry, schema_dir)

    if not registry.has(name):
        console.print(f"[red]Action type not found: {name}[/red]")
        sys.exit(1)

    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON params: {exc}[/red]")
        sys.exit(1)

    audit = AuditTrail()
    executor = ActionExecutor(registry, audit)

    # Validate only (no handler registered = execution will fail with a clear message)
    validation = executor.validate(name, parsed_params)
    if not validation.valid:
        console.print(f"[red]✗ Validation failed:[/red]")
        for err in validation.errors:
            console.print(f"  • {err}")
        sys.exit(1)

    console.print(f"[green]✓[/green] Params validated for action '{name}'")
    console.print(f"  Actor: {actor}")
    console.print(f"  Params: {json.dumps(parsed_params, ensure_ascii=False)}")
    console.print(f"\n[yellow]Note:[/yellow] No handler registered. "
                  "To execute, register a handler in your application code.")


@action.command("validate")
@click.argument("name")
@click.option("--params", "-p", required=True, help="JSON string of action parameters")
@click.option("--schema-dir", default=None, help="Directory containing action YAML files")
def action_validate(name: str, params: str, schema_dir: str | None) -> None:
    """Validate parameters against an action type's input schema."""
    from ontology_engine.kinetic.action_types import ActionRegistry

    registry = ActionRegistry()
    _load_actions_into_registry(registry, schema_dir)

    if not registry.has(name):
        console.print(f"[red]Action type not found: {name}[/red]")
        sys.exit(1)

    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON params: {exc}[/red]")
        sys.exit(1)

    result = registry.validate_input(name, parsed_params)
    if result.valid:
        console.print(f"[green]✓[/green] Parameters are valid for action '{name}'")
    else:
        console.print(f"[red]✗[/red] Validation errors:")
        for err in result.errors:
            console.print(f"  • {err}")
        sys.exit(1)


@main.group()
def audit() -> None:
    """Query the Kinetic Layer audit trail."""
    pass


@audit.command("query")
@click.option("--action", "action_name", default=None, help="Filter by action name")
@click.option("--actor", default=None, help="Filter by actor")
@click.option("--status", default=None, help="Filter by status")
@click.option("--limit", "-n", default=20, help="Max results")
def audit_query(action_name: str | None, actor: str | None, status: str | None, limit: int) -> None:
    """Query audit trail entries (in-memory only; use API for PG backend)."""
    console.print("[dim]Audit trail queries require a running application context.[/dim]")
    console.print("[dim]Use the SDK or API to query the audit trail:[/dim]")
    console.print()
    console.print("  # SDK usage:")
    console.print("  entries = await client.get_audit_trail(action_name='...', limit=20)")
    console.print()
    console.print("  # API usage:")
    console.print("  GET /api/v1/audit?action_name=...&limit=20")


def _load_actions_into_registry(
    registry: "ActionRegistry",
    schema_dir: str | None = None,
) -> None:
    """Load action definitions from YAML files into the registry."""
    import yaml

    from ontology_engine.kinetic.action_types import load_actions_from_yaml

    search_dirs: list[Path] = []
    if schema_dir:
        search_dirs.append(Path(schema_dir))
    # Also search default locations
    search_dirs.append(_find_schema_dir() / "examples")
    search_dirs.append(_find_schema_dir())

    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for yaml_file in sorted(sdir.glob("actions*.yaml")) + sorted(sdir.glob("actions*.yml")):
            try:
                raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if raw and isinstance(raw, dict):
                    for action_type in load_actions_from_yaml(raw):
                        registry.register(action_type)
            except Exception as exc:
                console.print(f"[yellow]Warning: Failed to load {yaml_file}: {exc}[/yellow]")


if __name__ == "__main__":
    main()
