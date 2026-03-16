"""FastAPI application factory for the Ontology Engine HTTP API.

Usage:
    app = create_app(db_url="postgresql://...", schema="ontology")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app(
    db_url: str | None = None, schema: str = "ontology",
    title: str = "Ontology Engine API", version: str = "0.1.0",
) -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Install with: pip install ontology-engine[api]")

    from ontology_engine.api.routes import router, set_client
    from ontology_engine.sdk.client import OntologyClient

    resolved_url = db_url or os.environ.get("DATABASE_URL", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if resolved_url:
            client = OntologyClient(db_url=resolved_url, schema=schema)
            await client.connect()
            set_client(client)
            logger.info("OntologyClient connected (schema=%s)", schema)
            yield
            await client.close()
        else:
            logger.warning("No DATABASE_URL — API running without DB")
            yield

    app = FastAPI(title=title, version=version,
                  description="HTTP API for the Ontology Engine knowledge graph",
                  lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    return app


def main() -> None:
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Ontology Engine HTTP API")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL", ""))
    parser.add_argument("--schema", default="ontology")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    app = create_app(db_url=args.db_url, schema=args.schema)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
