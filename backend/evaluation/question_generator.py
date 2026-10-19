"""50 Landmark Evaluation Questions with Scope Categorization (Global, Intermediate, Local)."""
from typing import List, Dict, Any
from pydantic import BaseModel

class EvalQuestion(BaseModel):
    id: str
    paper_id: str
    paper_title: str
    category: str # "global", "intermediate", "local"
    question: str
    ground_truth_hint: str

EVAL_QUESTIONS: List[EvalQuestion] = [
    # --- 15 Global / Thematic Questions ---
    EvalQuestion(
        id="q_g_01",
        paper_id="1706.03762_Attention_Is_All_You_Need",
        paper_title="Attention Is All You Need",
        category="global",
        question="How does the Transformer architecture replace recurrence with self-attention, and what theoretical and computational advantages does this provide?",
        ground_truth_hint="Eliminates recurrence for parallelization; O(1) path length between distant tokens versus O(n) in RNNs."
    ),
    EvalQuestion(
        id="q_g_02",
        paper_id="1609.02907_GCN",
        paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
        category="global",
        question="What is the foundational theoretical derivation of spectral graph convolutions and how does GCN simplify it to a first-order localized approximation?",
        ground_truth_hint="Approximates Chebyshev polynomials with K=1, lambda_max=2, renormalization trick adding self-loops."
    ),
    EvalQuestion(
        id="q_g_03",
        paper_id="1710.10903_GAT",
        paper_title="Graph Attention Networks",
        category="global",
        question="How do Graph Attention Networks overcome the fixed-weight limitation of isotropic GCNs through multi-head self-attention mechanisms?",
        ground_truth_hint="Computes anisotropic attention coefficients over neighbors without requiring costly matrix inversions."
    ),
    EvalQuestion(
        id="q_g_04",
        paper_id="1706.02216_GraphSAGE",
        paper_title="Inductive Representation Learning on Large Graphs",
        category="global",
        question="How does GraphSAGE transition from transductive graph representation learning to inductive batch sampling for unseen nodes?",
        ground_truth_hint="Learns aggregator functions (Mean, LSTM, Pooling) over uniform neighbor samplings rather than distinct embeddings."
    ),
    EvalQuestion(
        id="q_g_05",
        paper_id="2005.11401_RAG",
        paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        category="global",
        question="Compare RAG-Sequence and RAG-Token models in terms of latent document marginalization and generation flexibility.",
        ground_truth_hint="RAG-Sequence treats the retrieved document as a single latent variable; RAG-Token marginalizes per generated token."
    ),
    EvalQuestion(
        id="q_g_06",
        paper_id="1403.6652_DeepWalk",
        paper_title="DeepWalk: Online Learning of Social Representations",
        category="global",
        question="How does DeepWalk generalize language modeling representations (Word2Vec) to truncated uniform random walks on graphs?",
        ground_truth_hint="Treats random walk sequences as sentences and nodes as words obeying power-law frequencies."
    ),
    EvalQuestion(
        id="q_g_07",
        paper_id="1607.00653_node2vec",
        paper_title="node2vec: Scalable Feature Learning for Networks",
        category="global",
        question="Explain how the return parameter p and in-out parameter q smoothly interpolate between BFS and DFS exploration in node2vec.",
        ground_truth_hint="High p discourages backtracking; low q biases walk towards undiscovered nodes (DFS for structural equivalence)."
    ),
    EvalQuestion(
        id="q_g_08",
        paper_id="1810.04805_BERT",
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
        category="global",
        question="Why is deep bidirectionality in BERT critical compared to unidirectional LSTMs, and how does the Masked LM objective prevent label leakage?",
        ground_truth_hint="Masked LM corrupts 15% of tokens with [MASK] allowing tokens to attend bidirectionally without trivially seeing targets."
    ),
    EvalQuestion(
        id="q_g_09",
        paper_id="1704.01212_Neural_Message_Passing_Quantum_Chemistry",
        paper_title="Neural Message Passing for Quantum Chemistry",
        category="global",
        question="What are the three unified phases of Message Passing Neural Networks (MPNN) and how do they formalize invariance properties?",
        ground_truth_hint="Message function, Node update function, and Readout function; ensures invariance to graph isomorphisms."
    ),
    EvalQuestion(
        id="q_g_10",
        paper_id="1905.13192_RoBERTa",
        paper_title="RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        category="global",
        question="What design choices in the original BERT pre-training were shown to be undertrained or counterproductive in RoBERTa?",
        ground_truth_hint="Removing Next Sentence Prediction (NSP), dynamic masking, larger mini-batch sizes (8k), and training longer on 160GB text."
    ),
    EvalQuestion(
        id="q_g_11",
        paper_id="1706.03762_Attention_Is_All_You_Need",
        paper_title="Attention Is All You Need",
        category="global",
        question="How do positional encodings allow a non-recurrent Transformer to retain sequence ordering information across layers?",
        ground_truth_hint="Sinusoidal wave functions of varying frequencies added to input embeddings: PE(pos, 2i) = sin(pos/10000^(2i/d_model))."
    ),
    EvalQuestion(
        id="q_g_12",
        paper_id="1609.02907_GCN",
        paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
        category="global",
        question="Why is semi-supervised learning particularly effective when combined with graph convolutions on citation networks?",
        ground_truth_hint="Neighborhood homophily propagates sparse label gradients smoothly over localized graph Laplacians."
    ),
    EvalQuestion(
        id="q_g_13",
        paper_id="2005.11401_RAG",
        paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        category="global",
        question="How does retrieval-augmented generation mitigate hallucination and parameter bloat in large autoregressive language models?",
        ground_truth_hint="Decouples non-parametric memory (dense vector index of Wikipedia) from parametric model weights."
    ),
    EvalQuestion(
        id="q_g_14",
        paper_id="1706.02216_GraphSAGE",
        paper_title="Inductive Representation Learning on Large Graphs",
        category="global",
        question="How do pooling aggregators in GraphSAGE capture structural features independently of neighbor permutation?",
        ground_truth_hint="Applies a multi-layer perceptron to neighbor vectors followed by a symmetric element-wise max-pooling operator."
    ),
    EvalQuestion(
        id="q_g_15",
        paper_id="1810.04805_BERT",
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
        category="global",
        question="How does fine-tuning BERT on downstream sentence-pair classification tasks differ from pre-training?",
        ground_truth_hint="Uses the pooled representation of the first token [CLS] fed into a linear classification layer with minimal extra parameters."
    ),

    # --- 20 Intermediate / Community-Level Questions ---
    EvalQuestion(
        id="q_i_01",
        paper_id="1706.03762_Attention_Is_All_You_Need",
        paper_title="Attention Is All You Need",
        category="intermediate",
        question="Explain the role and formula of Scaled Dot-Product Attention, particularly the necessity of dividing by sqrt(d_k).",
        ground_truth_hint="Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V; prevents dot products from growing large in magnitude pushing softmax into small gradients."
    ),
    EvalQuestion(
        id="q_i_02",
        paper_id="1706.03762_Attention_Is_All_You_Need",
        paper_title="Attention Is All You Need",
        category="intermediate",
        question="Describe the structure of each Transformer decoder sub-layer and how masked attention preserves autoregressive causality.",
        ground_truth_hint="Masked self-attention sets illegal future connections to -inf before softmax; followed by encoder-decoder cross-attention and FFN."
    ),
    EvalQuestion(
        id="q_i_03",
        paper_id="1609.02907_GCN",
        paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
        category="intermediate",
        question="What is the renormalization trick in GCN, and why does unrenormalized (A + I_N) cause exploding or vanishing gradients in deep stacks?",
        ground_truth_hint="Computes D~^(-1/2) A~ D~^(-1/2) where A~ = A + I; normalizes eigenvalues to prevent scale blowup during repeated multiplications."
    ),
    EvalQuestion(
        id="q_i_04",
        paper_id="1710.10903_GAT",
        paper_title="Graph Attention Networks",
        category="intermediate",
        question="How does multi-head attention in GAT compute normalized coefficients alpha_ij using LeakyReLU non-linearities?",
        ground_truth_hint="alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j])); heads are concatenated for hidden layers and averaged for the output layer."
    ),
    EvalQuestion(
        id="q_i_05",
        paper_id="1706.02216_GraphSAGE",
        paper_title="Inductive Representation Learning on Large Graphs",
        category="intermediate",
        question="What aggregator architectures were evaluated in GraphSAGE, and which aggregator performed consistently best across benchmarks?",
        ground_truth_hint="Evaluated Mean, LSTM (on random permutations), and Max-Pooling; Max-Pooling aggregator achieved highest F1 scores."
    ),
    EvalQuestion(
        id="q_i_06",
        paper_id="2005.11401_RAG",
        paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        category="intermediate",
        question="How does the Dense Passage Retriever (DPR) bi-encoder interact with the BART generator during RAG training?",
        ground_truth_hint="DPR scores passages using query and doc BERT encoders; top-k passages are passed to BART cross-attention with joint backpropagation."
    ),
    EvalQuestion(
        id="q_i_07",
        paper_id="1403.6652_DeepWalk",
        paper_title="DeepWalk: Online Learning of Social Representations",
        category="intermediate",
        question="How does Hierarchical Softmax accelerate vertex prediction in DeepWalk from O(|V|) to O(log |V|)?",
        ground_truth_hint="Assigns vertices to leaves of a binary tree, turning multi-class probability into a sequence of binary sigmoid path decisions."
    ),
    EvalQuestion(
        id="q_i_08",
        paper_id="1607.00653_node2vec",
        paper_title="node2vec: Scalable Feature Learning for Networks",
        category="intermediate",
        question="Describe the transition probability calculation pi_vx for a random walk traversing from node t to v and considering neighbor x.",
        ground_truth_hint="pi_vx = alpha_pq(t, x) * w_vx; alpha is 1/p if distance d_tx=0, 1 if d_tx=1, and 1/q if d_tx=2."
    ),
    EvalQuestion(
        id="q_i_09",
        paper_id="1810.04805_BERT",
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
        category="intermediate",
        question="Explain how BERT represents input tokens, segment embeddings, and positional embeddings in a single unified vector.",
        ground_truth_hint="Sums WordPiece token embedding + Segment embedding (Sentence A vs B) + Position embedding."
    ),
    EvalQuestion(
        id="q_i_10",
        paper_id="1704.01212_Neural_Message_Passing_Quantum_Chemistry",
        paper_title="Neural Message Passing for Quantum Chemistry",
        category="intermediate",
        question="How does the Readout function in MPNN ensure that molecular property predictions remain invariant to atom permutations?",
        ground_truth_hint="Computes y = R({h_v^T | v in V}) using symmetric set functions like gated sum or set2set recurrent pooling."
    ),
    EvalQuestion(
        id="q_i_11",
        paper_id="1905.13192_RoBERTa",
        paper_title="RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        category="intermediate",
        question="How does dynamic masking in RoBERTa improve over the static masking strategy used in the original BERT release?",
        ground_truth_hint="Generates the masking pattern every time a sequence is fed to the model, preventing the model from seeing identical masks across epochs."
    ),
    EvalQuestion(
        id="q_i_12",
        paper_id="1609.02907_GCN",
        paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
        category="intermediate",
        question="What layer-wise propagation rule is computed in GCN, and how are feature dimensions transformed between layers?",
        ground_truth_hint="H^(l+1) = sigma(D~^(-1/2) A~ D~^(-1/2) H^(l) W^(l)); transforms C-dimensional node features through weight matrix W."
    ),
    EvalQuestion(
        id="q_i_13",
        paper_id="1710.10903_GAT",
        paper_title="Graph Attention Networks",
        category="intermediate",
        question="How does GAT implement inductive capability on completely unseen graphs, such as protein-protein interaction (PPI) networks?",
        ground_truth_hint="Attention computation depends strictly on pairwise features of connected neighbors, not on global Laplacian eigenvectors."
    ),
    EvalQuestion(
        id="q_i_14",
        paper_id="1706.02216_GraphSAGE",
        paper_title="Inductive Representation Learning on Large Graphs",
        category="intermediate",
        question="How does GraphSAGE manage neighborhood explosion in deep multi-layer representations during training?",
        ground_truth_hint="Uniformly samples a fixed-size neighborhood (e.g. S1=25, S2=10) at each search depth instead of taking all neighbors."
    ),
    EvalQuestion(
        id="q_i_15",
        paper_id="2005.11401_RAG",
        paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        category="intermediate",
        question="What document index representation does RAG use for scalable MIPS (Maximum Inner Product Search)?",
        ground_truth_hint="FAISS Hierarchical Navigable Small World (HNSW) index over 21 million 100-word Wikipedia passages."
    ),
    EvalQuestion(
        id="q_i_16",
        paper_id="1403.6652_DeepWalk",
        paper_title="DeepWalk: Online Learning of Social Representations",
        category="intermediate",
        question="How does DeepWalk update vertex representations asynchronously across multi-core workers?",
        ground_truth_hint="Uses Asynchronous Stochastic Gradient Descent (Hogwild!) without locks over shared memory tables."
    ),
    EvalQuestion(
        id="q_i_17",
        paper_id="1607.00653_node2vec",
        paper_title="node2vec: Scalable Feature Learning for Networks",
        category="intermediate",
        question="What is the computational complexity of the 2nd-order random walk precomputations in node2vec?",
        ground_truth_hint="O(a * |E|) where a is average degree; alias sampling enables generating next walk step in O(1) time."
    ),
    EvalQuestion(
        id="q_i_18",
        paper_id="1810.04805_BERT",
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
        category="intermediate",
        question="What was the effect of Next Sentence Prediction (NSP) pre-training on Question Answering (SQuAD) and Natural Language Inference (MNLI)?",
        ground_truth_hint="Removing NSP caused significant drops in MNLI and SQuAD performance in the original BERT ablations."
    ),
    EvalQuestion(
        id="q_i_19",
        paper_id="1704.01212_Neural_Message_Passing_Quantum_Chemistry",
        paper_title="Neural Message Passing for Quantum Chemistry",
        category="intermediate",
        question="How are edge features incorporated into message passing functions in molecular property graphs?",
        ground_truth_hint="m_v^(t+1) = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_vw); e_vw encodes bond types (single, double, aromatic) and distances."
    ),
    EvalQuestion(
        id="q_i_20",
        paper_id="1905.13192_RoBERTa",
        paper_title="RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        category="intermediate",
        question="What tokenization vocabulary size and encoding did RoBERTa adopt compared to BERT?",
        ground_truth_hint="Adopted a 50,000 byte-level BPE vocabulary instead of BERT's 30,000 character-level WordPiece vocabulary."
    ),

    # --- 15 Local / Factual Questions ---
    EvalQuestion(
        id="q_l_01",
        paper_id="1706.03762_Attention_Is_All_You_Need",
        paper_title="Attention Is All You Need",
        category="local",
        question="What BLEU score did the Transformer (base) and Transformer (big) achieve on the WMT 2014 English-to-German translation task?",
        ground_truth_hint="Transformer base achieved 27.3 BLEU; Transformer big achieved 28.4 BLEU."
    ),
    EvalQuestion(
        id="q_l_02",
        paper_id="1706.03762_Attention_Is_All_You_Need",
        paper_title="Attention Is All You Need",
        category="local",
        question="What were the exact hyperparameter values for d_model, d_ff, and number of attention heads (h) in Transformer base?",
        ground_truth_hint="d_model = 512, d_ff = 2048, h = 8, d_k = d_v = 64."
    ),
    EvalQuestion(
        id="q_l_03",
        paper_id="1609.02907_GCN",
        paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
        category="local",
        question="What classification accuracy percentage did 2-layer GCN achieve on the Cora and Citeseer benchmark datasets?",
        ground_truth_hint="Cora: 81.5%; Citeseer: 70.3% (Pubmed: 79.0%)."
    ),
    EvalQuestion(
        id="q_l_04",
        paper_id="1609.02907_GCN",
        paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
        category="local",
        question="How many labeled nodes per class were used for training in the standard semi-supervised experimental setup on citation networks?",
        ground_truth_hint="20 labeled nodes per class (e.g. 140 labels for Cora's 7 classes)."
    ),
    EvalQuestion(
        id="q_l_05",
        paper_id="1710.10903_GAT",
        paper_title="Graph Attention Networks",
        category="local",
        question="What Micro-averaged F1 score did Graph Attention Networks achieve on the inductive PPI (Protein-Protein Interaction) dataset?",
        ground_truth_hint="Micro F1 of 97.3% on PPI test set."
    ),
    EvalQuestion(
        id="q_l_06",
        paper_id="1710.10903_GAT",
        paper_title="Graph Attention Networks",
        category="local",
        question="How many attention heads and hidden features were configured in GAT for the Cora citation classification task?",
        ground_truth_hint="First layer: K=8 heads with F'=8 features each; second layer: single head with C=7 classes."
    ),
    EvalQuestion(
        id="q_l_07",
        paper_id="1706.02216_GraphSAGE",
        paper_title="Inductive Representation Learning on Large Graphs",
        category="local",
        question="What neighborhood sampling sizes (S_k) were chosen for depth K=1 and depth K=2 in GraphSAGE?",
        ground_truth_hint="S_1 = 25 for first hop, S_2 = 10 for second hop."
    ),
    EvalQuestion(
        id="q_l_08",
        paper_id="2005.11401_RAG",
        paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        category="local",
        question="How many total Wikipedia passages and chunks are indexed in the non-parametric memory of RAG?",
        ground_truth_hint="21,015,324 passages of 100 words each from Dec 2018 Wikipedia dump."
    ),
    EvalQuestion(
        id="q_l_09",
        paper_id="2005.11401_RAG",
        paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        category="local",
        question="What Exact Match (EM) score did RAG-Token achieve on the Natural Questions (NQ) open-domain benchmark?",
        ground_truth_hint="44.1% Exact Match on Natural Questions test set."
    ),
    EvalQuestion(
        id="q_l_10",
        paper_id="1810.04805_BERT",
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
        category="local",
        question="What are the layer counts (L), hidden sizes (H), and total parameters of BERT_BASE and BERT_LARGE?",
        ground_truth_hint="BERT_BASE: L=12, H=768, A=12, 110M params; BERT_LARGE: L=24, H=1024, A=16, 340M params."
    ),
    EvalQuestion(
        id="q_l_11",
        paper_id="1810.04805_BERT",
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
        category="local",
        question="What GLUE benchmark average score did BERT_LARGE achieve compared to previous SOTA OpenAI GPT?",
        ground_truth_hint="BERT_LARGE achieved 80.5 on GLUE, outperforming OpenAI GPT (72.8) by 7.7 points."
    ),
    EvalQuestion(
        id="q_l_12",
        paper_id="1905.13192_RoBERTa",
        paper_title="RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        category="local",
        question="What was the total pre-training dataset size in gigabytes used by RoBERTa compared to original BERT's 16GB?",
        ground_truth_hint="160 GB of uncompressed text (BooksCorpus, English Wikipedia, CC-News, OpenWebText, Stories)."
    ),
    EvalQuestion(
        id="q_l_13",
        paper_id="1403.6652_DeepWalk",
        paper_title="DeepWalk: Online Learning of Social Representations",
        category="local",
        question="What walk length (t), number of walks per vertex (gamma), and window size (w) were used in standard DeepWalk experiments?",
        ground_truth_hint="Walk length t = 40, gamma = 80 walks per vertex, window size w = 10."
    ),
    EvalQuestion(
        id="q_l_14",
        paper_id="1607.00653_node2vec",
        paper_title="node2vec: Scalable Feature Learning for Networks",
        category="local",
        question="What Macro-F1 score did node2vec achieve on the BlogCatalog network with 10% labeled data?",
        ground_truth_hint="Macro-F1 of ~0.27 with 10% labels (0.41 with 50% labels), outperforming DeepWalk and Line."
    ),
    EvalQuestion(
        id="q_l_15",
        paper_id="1704.01212_Neural_Message_Passing_Quantum_Chemistry",
        paper_title="Neural Message Passing for Quantum Chemistry",
        category="local",
        question="On the QM9 chemical benchmark, how many quantum mechanical properties was MPNN able to predict within chemical accuracy?",
        ground_truth_hint="Achieved chemical accuracy on 11 out of 13 molecular properties."
    ),
]

def get_evaluation_questions(category: str = "all", limit: int = 50) -> List[EvalQuestion]:
    """Retrieve filtered evaluation questions."""
    if category.lower() != "all":
        filtered = [q for q in EVAL_QUESTIONS if q.category.lower() == category.lower()]
    else:
        filtered = EVAL_QUESTIONS
    return filtered[:limit]
