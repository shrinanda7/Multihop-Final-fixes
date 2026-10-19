"""Preloader script to fetch 10 landmark AI & Graph research papers from arXiv."""
import os
import sys
import time
import logging
from pathlib import Path
import httpx

# Ensure backend path is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preload")

PAPERS = [
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "filename": "1706.03762_Attention_Is_All_You_Need.pdf"
    },
    {
        "id": "1609.02907",
        "title": "Semi-Supervised Classification with Graph Convolutional Networks",
        "filename": "1609.02907_GCN.pdf"
    },
    {
        "id": "1710.10903",
        "title": "Graph Attention Networks",
        "filename": "1710.10903_GAT.pdf"
    },
    {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "filename": "2005.11401_RAG.pdf"
    },
    {
        "id": "1706.02216",
        "title": "Inductive Representation Learning on Large Graphs",
        "filename": "1706.02216_GraphSAGE.pdf"
    },
    {
        "id": "1403.6652",
        "title": "DeepWalk: Online Learning of Social Representations",
        "filename": "1403.6652_DeepWalk.pdf"
    },
    {
        "id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "filename": "1810.04805_BERT.pdf"
    },
    {
        "id": "1607.00653",
        "title": "node2vec: Scalable Feature Learning for Networks",
        "filename": "1607.00653_node2vec.pdf"
    },
    {
        "id": "2201.11903",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "filename": "2201.11903_Chain_of_Thought.pdf"
    },
    {
        "id": "1703.08098",
        "title": "Knowledge Graph Embedding: A Survey of Approaches and Applications",
        "filename": "1703.08098_KG_Embedding_Survey.pdf"
    }
]

def download_papers(target_dir: Path = settings.PAPERS_DIR):
    """Download the 10 landmark papers from arXiv."""
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory for papers: {target_dir}")

    client = httpx.Client(timeout=60.0, follow_redirects=True, headers={"User-Agent": "ResearchGraphRAG/1.0 (academic-research)"})

    downloaded = 0
    for idx, paper in enumerate(PAPERS, 1):
        dest = target_dir / paper["filename"]
        if dest.exists() and dest.stat().st_size > 1000:
            logger.info(f"[{idx}/10] Already exists: {paper['title']} -> {dest.name} ({dest.stat().st_size // 1024} KB)")
            downloaded += 1
            continue

        url = f"https://arxiv.org/pdf/{paper['id']}.pdf"
        logger.info(f"[{idx}/10] Fetching arXiv:{paper['id']} ('{paper['title']}')...")

        try:
            res = client.get(url)
            if res.status_code == 200 and len(res.content) > 1000:
                with open(dest, "wb") as f:
                    f.write(res.content)
                logger.info(f"  ✓ Saved to {dest.name} ({len(res.content) // 1024} KB)")
                downloaded += 1
            else:
                logger.warning(f"  ✗ Failed to download {url}: HTTP {res.status_code}")
        except Exception as e:
            logger.error(f"  ✗ Exception while downloading {url}: {e}")

        time.sleep(1.0) # Polite pause between arXiv requests

    logger.info(f"\nFinished preloading papers. Total available: {downloaded}/{len(PAPERS)}")
    return downloaded

if __name__ == "__main__":
    download_papers()
