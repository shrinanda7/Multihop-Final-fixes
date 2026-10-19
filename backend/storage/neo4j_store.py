"""Neo4j Storage module re-exporting Neo4jStore, singleton neo4j_store, and persistence results."""
from backend.pipeline.neo4j_store import (
    Neo4jStore,
    neo4j_store,
    PersistenceResult,
    PersistenceError,
)

__all__ = ["Neo4jStore", "neo4j_store", "PersistenceResult", "PersistenceError"]
