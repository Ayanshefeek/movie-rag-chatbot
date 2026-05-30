import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from embed import EmbeddingPipeline


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.bm25 = None          # BM25 index
        self.corpus = []          # tokenized texts for BM25
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    # ─────────────────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────────────────

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self._build_bm25(metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    # ─────────────────────────────────────────────────────
    # BM25
    # ─────────────────────────────────────────────────────

    def _build_bm25(self, metadatas: List[Any]):
        print("[INFO] Building BM25 index...")
        self.corpus = [m["text"].lower().split() for m in metadatas]
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[INFO] BM25 index built with {len(self.corpus)} documents.")

    def _bm25_search(self, query_text: str, top_k: int = 5):
        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "bm25_score": float(scores[idx]),
                "metadata": self.metadata[idx] if idx < len(self.metadata) else None
            })
        return results

    # ─────────────────────────────────────────────────────
    # FAISS
    # ─────────────────────────────────────────────────────

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({
                "index": int(idx),
                "distance": float(dist),
                "metadata": meta
            })
        return results

    # ─────────────────────────────────────────────────────
    # HYBRID SEARCH
    # ─────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Hybrid querying for: '{query_text}'")

        # FAISS results
        query_emb = self.model.encode([query_text]).astype('float32')
        faiss_results = self.search(query_emb, top_k=top_k)

        # BM25 results
        bm25_results = self._bm25_search(query_text, top_k=top_k)

        # Merge using Reciprocal Rank Fusion (RRF)
        scores = {}

        for rank, r in enumerate(faiss_results):
            idx = r["index"]
            scores[idx] = scores.get(idx, 0) + 1 / (rank + 1)

        for rank, r in enumerate(bm25_results):
            idx = r["index"]
            scores[idx] = scores.get(idx, 0) + 1 / (rank + 1)

        # Sort by combined score
        sorted_indices = sorted(scores, key=scores.get, reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            results.append({
                "index": idx,
                "score": scores[idx],
                "metadata": self.metadata[idx] if idx < len(self.metadata) else None
            })

        return results

    # ─────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        bm25_path = os.path.join(self.persist_dir, "bm25.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus": self.corpus}, f)

        print(f"[INFO] Saved FAISS + BM25 index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        bm25_path = os.path.join(self.persist_dir, "bm25.pkl")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.corpus = data["corpus"]
            print(f"[INFO] Loaded FAISS + BM25 index and metadata from {self.persist_dir}")
        else:
            print("[WARN] BM25 index not found — rebuild the vector store to enable hybrid search.")

# ─────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("../data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("Christopher Nolan 2008", top_k=3))