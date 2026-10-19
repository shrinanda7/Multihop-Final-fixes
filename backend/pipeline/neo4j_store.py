"""Neo4j Graph Store for paper namespaces, entities, relations, chunks, and compressed views."""
import logging
from typing import Dict, Any, List, Optional, Set
try:
    from neo4j import GraphDatabase, Driver
except ImportError:  # Permit offline/unit-test mode without the optional database driver.
    GraphDatabase = None
    Driver = Any
from backend.config import settings

logger = logging.getLogger("graphrag.neo4j")

class PersistenceError(RuntimeError):
    """Raised when a graph persistence operation fails."""
    pass

class PersistenceResult(dict):
    """Explicit outcome of a graph persistence attempt.

    Subclasses dict for backwards compatibility with dictionary indexing while
    providing typed attribute access.
    """
    def __init__(
        self,
        success: bool,
        storage_mode: str,  # "neo4j", "in_memory", "failed"
        durable: bool,
        error: Optional[str] = None,
        message: Optional[str] = None,
        entities_count: int = 0,
        relations_count: int = 0,
        chunks_count: int = 0,
        **kwargs
    ):
        data = {
            "success": success,
            "storage_mode": storage_mode,
            "durable": durable,
            "error": error,
            "message": message,
            "entities_count": entities_count,
            "relations_count": relations_count,
            "chunks_count": chunks_count,
            **kwargs
        }
        super().__init__(data)
        self.success = success
        self.storage_mode = storage_mode
        self.durable = durable
        self.error = error
        self.message = message
        self.entities_count = entities_count
        self.relations_count = relations_count
        self.chunks_count = chunks_count

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        return f"PersistenceResult(success={self.success}, storage_mode='{self.storage_mode}', durable={self.durable}, error={self.error!r})"

class Neo4jStore:
    """Manages graph persistence in Neo4j with per-paper namespaces and dual original/compressed topologies."""

    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.username = settings.NEO4J_USERNAME
        self.password = settings.NEO4J_PASSWORD
        self.database = settings.NEO4J_DATABASE
        self._driver: Optional[Driver] = None
        self._connected = False
        self._in_memory_mode = False
        # In-memory graph store for fast querying or testing/offline fallback
        self._memory_nodes: Dict[str, List[Dict[str, Any]]] = {}
        self._memory_edges: Dict[str, List[Dict[str, Any]]] = {}
        self._memory_chunks: Dict[str, List[Dict[str, Any]]] = {}

    def set_in_memory_mode(self, enabled: bool = True):
        """Explicitly enable or disable intentional in-memory storage mode for testing or local dev."""
        self._in_memory_mode = enabled
        logger.info(f"Neo4jStore intentional in-memory mode set to: {enabled}")

    def get_driver(self) -> Optional[Driver]:
        """Lazy driver initialization with connection verification."""
        if self._in_memory_mode:
            return None

        if self._driver is None:
            if GraphDatabase is None or not self.uri or not self.username or not self.password:
                self._connected = False
                return None
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    connection_timeout=5.0,
                    max_connection_lifetime=300
                )
                # Verify connectivity
                with self._driver.session(database=self.database) as session:
                    session.run("RETURN 1 AS test").single()
                self._connected = True
                logger.info(f"Connected to Neo4j database '{self.database}' at {self.uri}")
            except Exception as e:
                logger.warning(f"Neo4j connection failed ({e}). Enabling in-memory fallback store.")
                self._driver = None
                self._connected = False
        return self._driver

    def check_health(self) -> Dict[str, Any]:
        """Health check reporting database status and stats."""
        try:
            if self._in_memory_mode:
                return {
                    "status": "in_memory_mode",
                    "uri": "memory://",
                    "database": "in_memory",
                    "using_fallback": True,
                    "storage_mode": "in_memory",
                    "durable": False,
                    "message": "Intentional in-memory development mode is enabled."
                }

            driver = self.get_driver()
            if driver and self._connected:
                with driver.session(database=self.database) as session:
                    result = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS version, edition").single()
                    version = result["version"] if result else "unknown"
                    counts = session.run("MATCH (n) RETURN count(n) AS node_count").single()
                    node_count = counts["node_count"] if counts else 0
                    return {
                        "status": "connected",
                        "uri": self.uri,
                        "database": self.database,
                        "version": version,
                        "total_nodes": node_count,
                        "using_fallback": False,
                        "storage_mode": "neo4j",
                        "durable": True
                    }
            return {
                "status": "offline",
                "uri": self.uri,
                "database": self.database,
                "using_fallback": True,
                "storage_mode": "in_memory",
                "durable": False,
                "message": "Neo4j Aura instance may still be resuming or credentials need review."
            }
        except Exception as e:
            return {
                "status": "error",
                "uri": self.uri,
                "error": str(e),
                "using_fallback": True,
                "storage_mode": "in_memory",
                "durable": False
            }

    def init_schema(self):
        """Create indexes and constraints for fast querying."""
        if self._in_memory_mode:
            return
        driver = self.get_driver()
        if not driver or not self._connected:
            return
        try:
            with driver.session(database=self.database) as session:
                session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.paper_id, e.id) IS UNIQUE")
                session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.paper_id, e.name)")
                session.run("CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.paper_id, c.chunk_id)")
                session.run("CREATE INDEX community_id_idx IF NOT EXISTS FOR (m:Community) ON (m.paper_id, m.id)")
        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j constraints: {e}")

    def save_entities_and_relations(
        self,
        paper_id: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        allow_memory_fallback: bool = True,
        raise_on_failure: bool = False
    ) -> PersistenceResult:
        """Persist extracted entities, directed relationships, and chunk links for a paper namespace.

        Uses an atomic Neo4j transaction unit to prevent partial writes.
        Returns an explicit PersistenceResult distinguishing:
        - durable Neo4j persistence (storage_mode='neo4j', durable=True)
        - in-memory fallback storage (storage_mode='in_memory', durable=False)
        - failed persistence (storage_mode='failed', durable=False)
        """
        # If in-memory mode is explicitly forced (for intentional testing/development)
        if self._in_memory_mode:
            self._memory_nodes[paper_id] = entities
            self._memory_edges[paper_id] = relations
            self._memory_chunks[paper_id] = chunks
            return PersistenceResult(
                success=True,
                storage_mode="in_memory",
                durable=False,
                entities_count=len(entities),
                relations_count=len(relations),
                chunks_count=len(chunks),
                message="Stored in intentional in-memory mode (not durable)."
            )

        driver = self.get_driver()
        if driver and self._connected:
            try:
                def _write_unit(tx):
                    # 1. Store chunks
                    chunk_query = """
                    UNWIND $chunks AS c
                    MERGE (chunk:Chunk {paper_id: $paper_id, chunk_id: c.chunk_id})
                    SET chunk.content = c.content,
                        chunk.section = c.section,
                        chunk.page = c.page
                    """
                    tx.run(chunk_query, chunks=chunks, paper_id=paper_id)

                    # 2. Store entities
                    entity_query = """
                    UNWIND $entities AS e
                    MERGE (ent:Entity {paper_id: $paper_id, id: e.id})
                    SET ent.name = e.name,
                        ent.type = e.type,
                        ent.description = e.description
                    WITH ent, e
                    UNWIND e.source_chunk_ids AS cid
                    MATCH (chunk:Chunk {paper_id: $paper_id, chunk_id: cid})
                    MERGE (ent)-[:MENTIONED_IN]->(chunk)
                    """
                    tx.run(entity_query, entities=entities, paper_id=paper_id)

                    # 3. Store directed relationships
                    relation_query = """
                    UNWIND $relations AS r
                    MATCH (source:Entity {paper_id: $paper_id, id: r.source})
                    MATCH (target:Entity {paper_id: $paper_id, id: r.target})
                    MERGE (source)-[rel:RELATED_TO {type: r.type}]->(target)
                    SET rel.description = r.description,
                        rel.weight = coalesce(r.weight, 1.0),
                        rel.source_chunk_ids = coalesce(r.source_chunk_ids, [])
                    """
                    tx.run(relation_query, relations=relations, paper_id=paper_id)

                with driver.session(database=self.database) as session:
                    session.execute_write(_write_unit)

                # Keep in-memory cache synchronized with durable state
                self._memory_nodes[paper_id] = entities
                self._memory_edges[paper_id] = relations
                self._memory_chunks[paper_id] = chunks

                return PersistenceResult(
                    success=True,
                    storage_mode="neo4j",
                    durable=True,
                    entities_count=len(entities),
                    relations_count=len(relations),
                    chunks_count=len(chunks),
                    message=f"Durably persisted {len(entities)} entities and {len(relations)} relations to Neo4j."
                )
            except Exception as e:
                logger.error(f"Error persisting graph to Neo4j for paper '{paper_id}': {e}", exc_info=True)
                # DO NOT silently store in memory or report success on write failure!
                res = PersistenceResult(
                    success=False,
                    storage_mode="failed",
                    durable=False,
                    error=str(e),
                    message=f"Neo4j write failed: {e}"
                )
                if raise_on_failure:
                    raise PersistenceError(res.error) from e
                return res

        # Driver is None or not connected (Neo4j unavailable/offline)
        if allow_memory_fallback:
            self._memory_nodes[paper_id] = entities
            self._memory_edges[paper_id] = relations
            self._memory_chunks[paper_id] = chunks
            return PersistenceResult(
                success=True,
                storage_mode="in_memory",
                durable=False,
                entities_count=len(entities),
                relations_count=len(relations),
                chunks_count=len(chunks),
                message="Stored in in-memory fallback store only (Neo4j offline/unavailable - not durably stored)."
            )
        else:
            res = PersistenceResult(
                success=False,
                storage_mode="failed",
                durable=False,
                error="Neo4j driver is unavailable and memory fallback is disabled.",
                message="Neo4j offline; memory fallback disabled."
            )
            if raise_on_failure:
                raise PersistenceError(res.error)
            return res

    def get_paper_graph(self, paper_id: str, compressed: bool = False) -> Dict[str, Any]:
        """Fetch either original or compressed graph topology for cytoscape visualization."""
        if not self._in_memory_mode:
            driver = self.get_driver()
            if driver and self._connected:
                try:
                    with driver.session(database=self.database) as session:
                        label = "Supernode" if compressed else "Entity"
                        node_result = session.run(
                            f"MATCH (n:{label} {{paper_id: $paper_id}}) RETURN properties(n) AS props",
                            paper_id=paper_id
                        )
                        nodes = [record["props"] for record in node_result]

                        edge_type = "COMPRESSED_EDGE" if compressed else "RELATED_TO"
                        edge_result = session.run(
                            f"""
                            MATCH (s:{label} {{paper_id: $paper_id}})-[r:{edge_type}]->(t:{label} {{paper_id: $paper_id}})
                            RETURN s.id AS source, t.id AS target, properties(r) AS props
                            """,
                            paper_id=paper_id
                        )
                        edges = [
                            {"source": r["source"], "target": r["target"], **r["props"]}
                            for r in edge_result
                        ]

                        entity_chunk_links = []
                        if not compressed:
                            link_result = session.run(
                                """
                                MATCH (e:Entity {paper_id: $paper_id})-[:MENTIONED_IN]->(c:Chunk {paper_id: $paper_id})
                                RETURN e.id AS entity_id, c.chunk_id AS chunk_id
                                """,
                                paper_id=paper_id
                            )
                            entity_chunk_links = [
                                {"entity_id": r["entity_id"], "chunk_id": r["chunk_id"]}
                                for r in link_result
                            ]

                        if nodes:
                            res = {
                                "nodes": nodes,
                                "edges": edges,
                                "compressed": compressed,
                                "storage_mode": "neo4j",
                                "durable": True
                            }
                            if not compressed:
                                links_by_entity = {}
                                for link in entity_chunk_links:
                                    links_by_entity.setdefault(str(link["entity_id"]), []).append(str(link["chunk_id"]))
                                for node in nodes:
                                    node["source_chunk_ids"] = sorted(set(node.get("source_chunk_ids", [])) | set(links_by_entity.get(str(node.get("id")), [])))
                                res["entity_chunk_links"] = entity_chunk_links
                            return res
                except Exception as e:
                    logger.warning(f"Neo4j query error: {e}. Falling back to memory.")

        # Fallback to in-memory representation
        nodes = self._memory_nodes.get(paper_id, [])
        edges = self._memory_edges.get(paper_id, [])
        return {
            "nodes": nodes,
            "edges": edges,
            "compressed": compressed,
            "storage_mode": "in_memory",
            "durable": False
        }

    def get_chunks_by_ids(
        self,
        paper_id: str,
        chunk_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve original text chunks with their stored provenance."""
        requested_ids = (
            list(dict.fromkeys(str(cid) for cid in chunk_ids))
            if chunk_ids
            else []
        )
        by_id: Dict[str, Dict[str, Any]] = {}

        if not self._in_memory_mode:
            driver = self.get_driver()
            if driver and self._connected:
                try:
                    with driver.session(database=self.database) as session:
                        if requested_ids:
                            query = """
                            MATCH (c:Chunk {paper_id: $paper_id})
                            WHERE c.chunk_id IN $chunk_ids
                            RETURN properties(c) AS props
                            """
                            result = session.run(
                                query,
                                paper_id=paper_id,
                                chunk_ids=requested_ids,
                            )
                        else:
                            query = """
                            MATCH (c:Chunk {paper_id: $paper_id})
                            RETURN properties(c) AS props
                            """
                            result = session.run(query, paper_id=paper_id)

                        for record in result:
                            chunk = dict(record["props"])
                            chunk_id = chunk.get("chunk_id")
                            if chunk_id is not None and chunk.get("content"):
                                by_id[str(chunk_id)] = chunk
                except Exception:
                    logger.exception(
                        "Failed to retrieve source chunks for paper %s",
                        paper_id,
                    )

        # Support in-memory fallback when the store retains original chunks.
        memory_chunks = getattr(self, "_memory_chunks", {})
        if isinstance(memory_chunks, dict):
            candidates = memory_chunks.get(paper_id, [])
        else:
            candidates = []

        if isinstance(candidates, dict):
            candidates = list(candidates.values())

        for chunk in candidates or []:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id")
            if (
                chunk_id is not None
                and (not requested_ids or str(chunk_id) in requested_ids)
                and chunk.get("content")
            ):
                by_id.setdefault(str(chunk_id), dict(chunk))

        if requested_ids:
            return [
                by_id[cid]
                for cid in requested_ids
                if cid in by_id
            ]
        return sorted(by_id.values(), key=lambda c: str(c.get("chunk_id", "")))

    def clear_paper_graph(self, paper_id: str) -> Dict[str, Any]:
        """Delete all entities, relations, chunks, and views for a paper namespace."""
        self._memory_nodes.pop(paper_id, None)
        self._memory_edges.pop(paper_id, None)
        self._memory_chunks.pop(paper_id, None)

        if not self._in_memory_mode:
            driver = self.get_driver()
            if driver and self._connected:
                try:
                    with driver.session(database=self.database) as session:
                        prefix = f"{paper_id}_"
                        del_q = """
                        MATCH (n)
                        WHERE n.paper_id = $paper_id
                           OR (n.id IS NOT NULL AND (n.id = $paper_id OR n.id STARTS WITH $prefix))
                           OR (n.chunk_id IS NOT NULL AND n.chunk_id STARTS WITH $prefix)
                           OR (n.name IS NOT NULL AND n.name = $paper_id)
                        DETACH DELETE n
                        """
                        session.run(del_q, paper_id=paper_id, prefix=prefix)
                    return {"success": True, "storage_mode": "neo4j"}
                except Exception as e:
                    logger.error(f"Error clearing Neo4j paper {paper_id}: {e}")
                    return {"success": False, "error": str(e)}

        return {"success": True, "storage_mode": "in_memory"}

    def clear_all_graphs(self) -> Dict[str, Any]:
        """Purge all nodes and relationships from Neo4j Aura and memory."""
        self._memory_nodes.clear()
        self._memory_edges.clear()
        self._memory_chunks.clear()

        driver = self.get_driver()
        if driver and self._connected:
            try:
                with driver.session(database=self.database) as session:
                    res = session.run("MATCH (n) DETACH DELETE n RETURN count(n) as deleted")
                    cnt = res.single()["deleted"]
                return {"success": True, "nodes_deleted": cnt}
            except Exception as e:
                logger.error(f"Error purging all Neo4j nodes: {e}")
                return {"success": False, "error": str(e)}
        return {"success": True, "nodes_deleted": 0}

    def close(self):
        if self._driver:
            self._driver.close()

# Singleton instance
neo4j_store = Neo4jStore()
