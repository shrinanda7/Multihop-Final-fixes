import { Paper, GraphTopology, ConfigState, BenchmarkConfigSummary } from "./types";
import { getPaperGraphData, getMultiPaperGraphData } from "./paperGraphs";

export const INITIAL_CONFIG: ConfigState = {
  ENABLE_TARJAN_COMPRESSION: true,
  ENABLE_BRIDGE_AWARE_PRUNING: true,
  ENABLE_BUDGET_AWARE_TRAVERSAL: true,
  GIANT_SCC_THRESHOLD: 0.30,
  RELEVANCE_THRESHOLD: 60.0,
  TOTAL_LLM_BUDGET: 12,
  MAX_COMMUNITY_DEPTH: 2,
  OLLAMA_SMALL_MODEL: "llama3.2:3b",
  OLLAMA_LARGE_MODEL: "llama3.1:8b"
};

export const PRELOADED_PAPERS: Paper[] = [
  {
    paper_id: "1706.03762_Attention_Is_All_You_Need",
    title: "Attention Is All You Need",
    filename: "1706.03762_Attention_Is_All_You_Need.pdf",
    file_size_kb: 2160,
    is_preloaded: true,
    status: "ready",
    num_entities: 18,
    num_relations: 26,
    num_communities: 4
  },
  {
    paper_id: "1609.02907_GCN",
    title: "Semi-Supervised Classification with Graph Convolutional Networks",
    filename: "1609.02907_GCN.pdf",
    file_size_kb: 1480,
    is_preloaded: true,
    status: "ready",
    num_entities: 16,
    num_relations: 24,
    num_communities: 4
  },
  {
    paper_id: "1710.10903_GAT",
    title: "Graph Attention Networks",
    filename: "1710.10903_GAT.pdf",
    file_size_kb: 1820,
    is_preloaded: true,
    status: "ready",
    num_entities: 15,
    num_relations: 21,
    num_communities: 3
  },
  {
    paper_id: "1706.02216_GraphSAGE",
    title: "Inductive Representation Learning on Large Graphs",
    filename: "1706.02216_GraphSAGE.pdf",
    file_size_kb: 1940,
    is_preloaded: true,
    status: "ready",
    num_entities: 17,
    num_relations: 25,
    num_communities: 4
  },
  {
    paper_id: "2005.11401_RAG",
    title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    filename: "2005.11401_RAG.pdf",
    file_size_kb: 2310,
    is_preloaded: true,
    status: "ready",
    num_entities: 19,
    num_relations: 27,
    num_communities: 5
  },
  {
    paper_id: "1403.6652_DeepWalk",
    title: "DeepWalk: Online Learning of Social Representations",
    filename: "1403.6652_DeepWalk.pdf",
    file_size_kb: 1540,
    is_preloaded: true,
    status: "ready",
    num_entities: 14,
    num_relations: 19,
    num_communities: 3
  },
  {
    paper_id: "1607.00653_node2vec",
    title: "node2vec: Scalable Feature Learning for Networks",
    filename: "1607.00653_node2vec.pdf",
    file_size_kb: 1680,
    is_preloaded: true,
    status: "ready",
    num_entities: 16,
    num_relations: 22,
    num_communities: 4
  },
  {
    paper_id: "1810.04805_BERT",
    title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    filename: "1810.04805_BERT.pdf",
    file_size_kb: 2890,
    is_preloaded: true,
    status: "ready",
    num_entities: 20,
    num_relations: 29,
    num_communities: 5
  },
  {
    paper_id: "2201.11903_Chain_of_Thought",
    title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    filename: "2201.11903_Chain_of_Thought.pdf",
    file_size_kb: 1720,
    is_preloaded: true,
    status: "ready",
    num_entities: 15,
    num_relations: 20,
    num_communities: 3
  },
  {
    paper_id: "1703.08098_KG_Embedding_Survey",
    title: "Knowledge Graph Embedding: A Survey of Approaches and Applications",
    filename: "1703.08098_KG_Embedding_Survey.pdf",
    file_size_kb: 2450,
    is_preloaded: true,
    status: "ready",
    num_entities: 22,
    num_relations: 31,
    num_communities: 6
  }
];

export function generatePaperGraph(paperId: string, customTitle?: string): { original_graph: GraphTopology; compressed_graph: GraphTopology } {
  return getPaperGraphData(paperId, customTitle);
}

export function generateMultiPaperGraph(paperIds: string[]): { original_graph: GraphTopology; compressed_graph: GraphTopology } {
  return getMultiPaperGraphData(paperIds);
}

export const BENCHMARK_CONFIG_SUMMARIES: BenchmarkConfigSummary[] = [
  {
    configuration: "Ours (Full: Tarjan + Bridge-Aware + Budget)",
    avg_score: 4.62,
    avg_tokens: 2840,
    avg_llm_calls: 3.4,
    avg_latency_ms: 412.5,
    compression_ratio: 0.28,
    graph_preservation: 0.985,
    evaluated_samples: 50
  },
  {
    configuration: "Ablation: No Tarjan Compression",
    avg_score: 4.31,
    avg_tokens: 5890,
    avg_llm_calls: 7.2,
    avg_latency_ms: 890.1,
    compression_ratio: 0.0,
    graph_preservation: 1.0,
    evaluated_samples: 50
  },
  {
    configuration: "Ablation: No Bridge-Aware Pruning",
    avg_score: 3.74,
    avg_tokens: 2410,
    avg_llm_calls: 2.8,
    avg_latency_ms: 385.0,
    compression_ratio: 0.35,
    graph_preservation: 0.724,
    evaluated_samples: 50
  },
  {
    configuration: "Ablation: No Budget-Aware Traversal",
    avg_score: 4.51,
    avg_tokens: 7210,
    avg_llm_calls: 8.9,
    avg_latency_ms: 1045.2,
    compression_ratio: 0.28,
    graph_preservation: 0.985,
    evaluated_samples: 50
  },
  {
    configuration: "Dynamic Traversal (Reference Baseline)",
    avg_score: 4.18,
    avg_tokens: 4120,
    avg_llm_calls: 5.1,
    avg_latency_ms: 620.4,
    compression_ratio: 0.15,
    graph_preservation: 0.812,
    evaluated_samples: 50
  },
  {
    configuration: "Static Map-Reduce (Level 1 Baseline)",
    avg_score: 3.82,
    avg_tokens: 6450,
    avg_llm_calls: 4.0,
    avg_latency_ms: 780.0,
    compression_ratio: 0.0,
    graph_preservation: 0.750,
    evaluated_samples: 50
  }
];
