export interface Paper {
  paper_id: string;
  title: string;
  filename: string;
  file_size_kb: number;
  is_preloaded: boolean;
  status: "ready" | "processing" | "pending" | "failed";
  num_entities: number;
  num_relations: number;
  num_communities?: number;
  content_hash?: string;
  summary?: string;
}

export interface TraversalStep {
  step_index: number;
  action: "VISIT" | "RATE" | "EXPAND" | "PRUNE" | "BRIDGE_PRESERVE" | "SELECT";
  node_or_comm_id: string;
  target_id?: string;
  level: number;
  score: number;
  reason: string;
  timestamp_ms: number;
}

export interface TraversalTrace {
  mode: string;
  steps: TraversalStep[];
  visited_nodes: string[];
  expanded_nodes: string[];
  pruned_nodes: string[];
  bridge_kept_nodes: string[];
  selected_reports: string[];
  ratings: Record<string, number>;
  total_tokens: number;
  total_llm_calls: number;
  latency_ms: number;
}

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  description?: string;
  is_supernode?: boolean;
  is_connector?: boolean;
  member_node_ids?: string[];
  member_nodes?: GraphNode[];
  member_count?: number;
  source_chunk_ids?: string[];
  paper_id?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  weight?: number;
  is_bridge?: boolean;
}

export interface SCCDiagnostics {
  total_nodes: number;
  num_sccs: number;
  scc_sizes: number[];
  max_scc_size: number;
  max_scc_ratio: number;
  has_giant_scc: boolean;
  mitigation_strategy?: string;
}

export interface GraphTopology {
  nodes: GraphNode[];
  edges: GraphEdge[];
  diagnostics?: SCCDiagnostics;
  compression_ratio?: number;
  articulation_points?: string[];
  bridges?: string[][];
  node_count?: number;
  edge_count?: number;
}

export interface CommunityInfo {
  id: string;
  level: number;
  title: string;
  member_node_ids: string[];
  parent_id?: string;
  children_ids?: string[];
  report?: string;
  summary?: string;
}

export interface QueryResult {
  mode: string;
  answer: string;
  trace: TraversalTrace;
  original_graph?: GraphTopology;
  compressed_graph?: GraphTopology;
  communities?: Record<string, CommunityInfo>;
  selected_reports?: string[];
  total_tokens: number;
  total_llm_calls: number;
  latency_ms: number;
  bridge_kept_count?: number;
  cited_passages?: { paper_title: string; text: string; score: number }[];
  papers_evaluated?: string[];
  engine?: string;
}

export interface ConfigState {
  ENABLE_TARJAN_COMPRESSION: boolean;
  ENABLE_BRIDGE_AWARE_PRUNING: boolean;
  ENABLE_BUDGET_AWARE_TRAVERSAL: boolean;
  GIANT_SCC_THRESHOLD: number;
  RELEVANCE_THRESHOLD: number;
  TOTAL_LLM_BUDGET: number;
  MAX_COMMUNITY_DEPTH: number;
  OLLAMA_SMALL_MODEL: string;
  OLLAMA_LARGE_MODEL: string;
}

export interface BenchmarkRecord {
  question_id: string;
  category: "global" | "intermediate" | "local";
  paper_id: string;
  configuration: string;
  mode: string;
  score: number;
  justification: string;
  total_tokens: number;
  total_llm_calls: number;
  latency_ms: number;
  compression_ratio: number;
  graph_preservation: number;
  bridge_kept_count: number;
}

export interface BenchmarkConfigSummary {
  configuration: string;
  avg_score: number;
  avg_tokens: number;
  avg_llm_calls: number;
  avg_latency_ms: number;
  compression_ratio: number;
  graph_preservation: number;
  evaluated_samples: number;
}

export interface Neo4jStatus {
  connected: boolean;
  status: "connected" | "offline" | "error";
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
}
