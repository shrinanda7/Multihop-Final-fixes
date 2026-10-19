import neo4j, { Driver, Session } from "neo4j-driver";
import { GraphTopology, GraphNode, GraphEdge } from "./types";

let driverInstance: Driver | null = null;

export function getNeo4jConfig() {
  return {
    // Credentials must come from the deployment environment; never hard-code secrets.
    uri: process.env.NEO4J_URI || "",
    username: process.env.NEO4J_USERNAME || "",
    password: process.env.NEO4J_PASSWORD || "",
    database: process.env.NEO4J_DATABASE || "neo4j",
    instanceId: process.env.AURA_INSTANCEID || "",
    instanceName: process.env.AURA_INSTANCENAME || "",
  };
}

export function getNeo4jDriver(): Driver | null {
  const config = getNeo4jConfig();
  if (!config.uri || !config.username || !config.password) {
    return null;
  }

  if (!driverInstance) {
    try {
      driverInstance = neo4j.driver(
        config.uri,
        neo4j.auth.basic(config.username, config.password),
        {
          maxConnectionLifetime: 3 * 60 * 1000,
          connectionTimeout: 10000,
        }
      );
    } catch (err) {
      console.error("[Neo4j Service] Error creating driver:", err);
      driverInstance = null;
    }
  }

  return driverInstance;
}

export async function checkNeo4jConnection(): Promise<{
  connected: boolean;
  status: "connected" | "offline";
  uri: string;
  database: string;
  username: string;
  instance_id: string;
  instance_name: string;
  version?: string;
  total_nodes?: number;
  total_relationships?: number;
  paper_count?: number;
  entity_count?: number;
  labels?: string[];
  error?: string;
}> {
  const config = getNeo4jConfig();
  const driver = getNeo4jDriver();

  if (!driver) {
    return {
      connected: false,
      status: "offline",
      uri: config.uri,
      database: config.database,
      username: config.username,
      instance_id: config.instanceId,
      instance_name: config.instanceName,
      error: "Driver not initialized. Check credentials.",
    };
  }

  let session: Session | null = null;
  try {
    session = driver.session({ database: config.database });
    await session.run("RETURN 1 AS ping");

    let version = "Neo4j Aura";
    try {
      const compRes = await session.run(
        "CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS version, edition"
      );
      if (compRes.records.length > 0) {
        version = `${compRes.records[0].get("name")} ${compRes.records[0].get("version")} (${compRes.records[0].get("edition")})`;
      }
    } catch {}

    const nodeCountRes = await session.run("MATCH (n) RETURN count(n) AS c");
    const total_nodes = Number(nodeCountRes.records[0]?.get("c") || 0);

    const relCountRes = await session.run("MATCH ()-[r]->() RETURN count(r) AS c");
    const total_relationships = Number(relCountRes.records[0]?.get("c") || 0);

    const paperCountRes = await session.run("MATCH (p:Paper) RETURN count(p) AS c");
    const paper_count = Number(paperCountRes.records[0]?.get("c") || 0);

    const entityCountRes = await session.run("MATCH (e:Entity) RETURN count(e) AS c");
    const entity_count = Number(entityCountRes.records[0]?.get("c") || 0);

    let labels: string[] = [];
    try {
      const labelsRes = await session.run("CALL db.labels() YIELD label RETURN label");
      labels = labelsRes.records.map((r) => r.get("label"));
    } catch {}

    return {
      connected: true,
      status: "connected",
      uri: config.uri,
      database: config.database,
      username: config.username,
      instance_id: config.instanceId,
      instance_name: config.instanceName,
      version,
      total_nodes,
      total_relationships,
      paper_count,
      entity_count,
      labels,
    };
  } catch (err: any) {
    return {
      connected: false,
      status: "offline",
      uri: config.uri,
      database: config.database,
      username: config.username,
      instance_id: config.instanceId,
      instance_name: config.instanceName,
      error: err?.message || "Connection failed",
    };
  } finally {
    if (session) await session.close();
  }
}

export async function initNeo4jConstraints(): Promise<void> {
  const driver = getNeo4jDriver();
  if (!driver) return;
  const config = getNeo4jConfig();
  const session = driver.session({ database: config.database });
  try {
    await session.run(
      "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE"
    );
    await session.run(
      "CREATE CONSTRAINT entity_paper_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.paper_id, e.id) IS UNIQUE"
    );
    await session.run(
      "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.paper_id, e.name)"
    );
  } catch (err: any) {
    console.warn("[Neo4j Service] Constraint setup warning:", err?.message);
  } finally {
    await session.close();
  }
}

export async function syncPaperToNeo4j(
  paperId: string,
  title: string,
  originalGraph: GraphTopology,
  compressedGraph?: GraphTopology
): Promise<{ success: boolean; nodes_synced: number; edges_synced: number; error?: string }> {
  const driver = getNeo4jDriver();
  if (!driver) return { success: false, nodes_synced: 0, edges_synced: 0, error: "No driver" };

  const config = getNeo4jConfig();
  const session = driver.session({ database: config.database });

  try {
    // 1. Upsert Paper node
    await session.run(
      `
      MERGE (p:Paper {paper_id: $paperId})
      SET p.title = $title,
          p.updated_at = datetime(),
          p.original_nodes = $origNodes,
          p.original_edges = $origEdges,
          p.compression_ratio = $compRatio
      `,
      {
        paperId,
        title,
        origNodes: originalGraph.nodes.length,
        origEdges: originalGraph.edges.length,
        compRatio: compressedGraph?.compression_ratio || 0,
      }
    );

    // 2. Batch upsert Entities
    const nodeBatch = originalGraph.nodes.map((n) => ({
      paper_id: paperId,
      id: n.id,
      name: n.name,
      type: n.type || "Concept",
      description: n.description || "",
      is_supernode: Boolean(n.is_supernode),
      is_connector: Boolean(n.is_connector),
    }));

    await session.run(
      `
      UNWIND $batch AS item
      MERGE (e:Entity {paper_id: item.paper_id, id: item.id})
      SET e.name = item.name,
          e.type = item.type,
          e.description = item.description,
          e.is_supernode = item.is_supernode,
          e.is_connector = item.is_connector,
          e.updated_at = datetime()
      WITH e, item
      MATCH (p:Paper {paper_id: item.paper_id})
      MERGE (p)-[:CONTAINS_ENTITY]->(e)
      `,
      { batch: nodeBatch }
    );

    // 3. Batch upsert Relationships
    const edgeBatch = originalGraph.edges.map((e) => ({
      paper_id: paperId,
      source: e.source,
      target: e.target,
      type: e.type || "RELATED_TO",
      is_bridge: Boolean(e.is_bridge),
      weight: e.weight ?? 1.0,
    }));

    await session.run(
      `
      UNWIND $batch AS edge
      MATCH (s:Entity {paper_id: edge.paper_id, id: edge.source})
      MATCH (t:Entity {paper_id: edge.paper_id, id: edge.target})
      MERGE (s)-[r:RELATION {type: edge.type}]->(t)
      SET r.is_bridge = edge.is_bridge,
          r.weight = edge.weight,
          r.paper_id = edge.paper_id,
          r.updated_at = datetime()
      `,
      { batch: edgeBatch }
    );

    // 4. If compressed supernodes exist, link them
    if (compressedGraph && compressedGraph.nodes) {
      const supernodeBatch = compressedGraph.nodes
        .filter((sn) => sn.is_supernode)
        .map((sn) => ({
          paper_id: paperId,
          id: sn.id,
          name: sn.name,
          member_count: sn.member_count || (sn.member_node_ids?.length ?? 1),
          member_node_ids: sn.member_node_ids || [],
        }));

      if (supernodeBatch.length > 0) {
        await session.run(
          `
          UNWIND $batch AS item
          MERGE (s:Supernode {paper_id: item.paper_id, id: item.id})
          SET s.name = item.name,
              s.member_count = item.member_count,
              s.member_node_ids = item.member_node_ids,
              s.updated_at = datetime()
          WITH s, item
          MATCH (p:Paper {paper_id: item.paper_id})
          MERGE (p)-[:HAS_SUPERNODE]->(s)
          `,
          { batch: supernodeBatch }
        );
      }
    }

    return {
      success: true,
      nodes_synced: originalGraph.nodes.length,
      edges_synced: originalGraph.edges.length,
    };
  } catch (err: any) {
    console.error(`[Neo4j Service] Error syncing paper ${paperId}:`, err);
    return {
      success: false,
      nodes_synced: 0,
      edges_synced: 0,
      error: err?.message || "Sync failed",
    };
  } finally {
    await session.close();
  }
}

export async function fetchPaperGraphFromNeo4j(
  paperId: string
): Promise<{ original_nodes: GraphNode[]; original_edges: GraphEdge[] } | null> {
  const driver = getNeo4jDriver();
  if (!driver) return null;

  const config = getNeo4jConfig();
  const session = driver.session({ database: config.database });

  try {
    const nodesRes = await session.run(
      `
      MATCH (e:Entity {paper_id: $paperId})
      RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description,
             e.is_supernode AS is_supernode, e.is_connector AS is_connector
      `,
      { paperId }
    );

    if (nodesRes.records.length === 0) {
      return null;
    }

    const original_nodes: GraphNode[] = nodesRes.records.map((r) => ({
      id: r.get("id"),
      name: r.get("name") || r.get("id"),
      type: r.get("type") || "Concept",
      description: r.get("description") || "",
      is_supernode: Boolean(r.get("is_supernode")),
      is_connector: Boolean(r.get("is_connector")),
    }));

    const edgesRes = await session.run(
      `
      MATCH (s:Entity {paper_id: $paperId})-[r:RELATION]->(t:Entity {paper_id: $paperId})
      RETURN s.id AS source, t.id AS target, r.type AS type, r.is_bridge AS is_bridge, r.weight AS weight
      `,
      { paperId }
    );

    const original_edges: GraphEdge[] = edgesRes.records.map((r) => ({
      source: r.get("source"),
      target: r.get("target"),
      type: r.get("type") || "RELATED_TO",
      is_bridge: Boolean(r.get("is_bridge")),
      weight: r.get("weight") ? Number(r.get("weight")) : 1.0,
    }));

    return { original_nodes, original_edges };
  } catch (err: any) {
    console.warn(`[Neo4j Service] Error retrieving paper graph for ${paperId}:`, err?.message);
    return null;
  } finally {
    await session.close();
  }
}

export async function runCypherQuery(
  cypher: string,
  params: Record<string, any> = {}
): Promise<{
  columns: string[];
  records: any[];
  execution_ms: number;
  summary: string;
}> {
  const start = Date.now();
  const driver = getNeo4jDriver();
  if (!driver) {
    throw new Error("Neo4j driver is not available");
  }

  const config = getNeo4jConfig();
  const session = driver.session({ database: config.database });

  try {
    const result = await session.run(cypher, params);
    const execution_ms = Date.now() - start;

    const columns: string[] = result.records.length > 0 ? result.records[0].keys.map((k) => String(k)) : [];
    const records = result.records.map((r) => {
      const obj: Record<string, any> = {};
      for (const rawK of r.keys) {
        const k = String(rawK);
        const val = r.get(k);
        // Handle Neo4j Integers or node objects
        if (val && typeof val === "object") {
          if ("low" in val && "high" in val) {
            obj[k] = val.low;
          } else if ("properties" in val) {
            obj[k] = val.properties;
          } else {
            obj[k] = val;
          }
        } else {
          obj[k] = val;
        }
      }
      return obj;
    });

    return {
      columns,
      records,
      execution_ms,
      summary: `Returned ${records.length} records in ${execution_ms}ms`,
    };
  } finally {
    await session.close();
  }
}

export async function deletePaperFromNeo4j(
  paperId: string
): Promise<{ success: boolean; nodes_deleted: number; error?: string }> {
  const driver = getNeo4jDriver();
  if (!driver) return { success: false, nodes_deleted: 0, error: "No driver" };

  const config = getNeo4jConfig();
  const session = driver.session({ database: config.database });

  try {
    const res = await session.run(
      `
      MATCH (n)
      WHERE n.paper_id = $paperId 
         OR toLower(n.paper_id) = toLower($paperId)
         OR (n.id IS NOT NULL AND (n.id = $paperId OR n.id STARTS WITH $paperPrefix))
         OR (n.chunk_id IS NOT NULL AND n.chunk_id STARTS WITH $paperPrefix)
         OR (n.name IS NOT NULL AND (n.name = $paperId OR toLower(n.name) = toLower($paperId)))
         OR (n.title IS NOT NULL AND (n.title = $paperId OR toLower(n.title) = toLower($paperId)))
      DETACH DELETE n
      RETURN count(n) AS totalDeleted
      `,
      { paperId, paperPrefix: `${paperId}_` }
    );

    const total = res.records[0]?.get("totalDeleted")?.low ?? 0;
    return { success: true, nodes_deleted: total };
  } catch (err: any) {
    console.warn(`[Neo4j Service] Delete error for paper ${paperId}:`, err);
    return { success: false, nodes_deleted: 0, error: err?.message };
  } finally {
    await session.close();
  }
}

export async function purgeAllFromNeo4j(): Promise<{
  success: boolean;
  nodes_deleted: number;
  error?: string;
}> {
  const driver = getNeo4jDriver();
  if (!driver) return { success: false, nodes_deleted: 0, error: "No driver" };

  const config = getNeo4jConfig();
  const session = driver.session({ database: config.database });

  try {
    const res = await session.run(`
      MATCH (n)
      DETACH DELETE n
      RETURN count(n) AS totalDeleted
    `);
    const total = res.records[0]?.get("totalDeleted")?.low ?? 0;
    console.log(`[Neo4j Service] Purged all ${total} nodes from Neo4j Aura.`);
    return { success: true, nodes_deleted: total };
  } catch (err: any) {
    console.error("[Neo4j Service] Error purging all nodes:", err);
    return { success: false, nodes_deleted: 0, error: err?.message };
  } finally {
    await session.close();
  }
}

