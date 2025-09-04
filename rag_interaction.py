import os
import json
import time
import re
from typing import List, Tuple, Dict, Optional

import requests
import numpy as np


class LocalRAG:
    """
    Class for a local RAG that uses Ollama for embeddings and generation.
    - ollama_url: Ollama base endpoint (e.g. http://localhost:11434/api)
    - embed_model: embedding model
    - chat_model: model for generation / chat
    - db_dir: folder to save the index (chunks.json, embeddings.npy, meta.json)
    """
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api",
        embed_model: str = "nomic-embed-text",
        chat_model: str = "gemma3:1b-it-qat",
        db_dir: str = "rag_db",
    ):
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.db_dir = db_dir

        # ensure folder exists
        os.makedirs(self.db_dir, exist_ok=True)

        # index file paths
        self.chunks_path = os.path.join(self.db_dir, "chunks.json")
        self.embeddings_path = os.path.join(self.db_dir, "embeddings.npy")
        self.meta_path = os.path.join(self.db_dir, "meta.json")

        # in-memory data
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    # Utilities
    def clear(self) -> None:
        """Clear the terminal (Windows/Unix)."""
        os.system("cls" if os.name == "nt" else "clear")

    # Embeddings (with fallback)
    def _request_embeddings(self, inputs: List[str]) -> List[List[float]]:

        # try batch request first
        try:
            payload = {"model": self.embed_model, "prompt": inputs}
            resp = requests.post(f"{self.ollama_url}/embeddings", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # possible formats: list of dicts or dict with 'embedding' -> array
            if isinstance(data, list):
                return [d["embedding"] if isinstance(d, dict) and "embedding" in d else d for d in data]
            if isinstance(data, dict) and "embedding" in data and isinstance(data["embedding"][0], list):
                return data["embedding"]
        except Exception:
            # if batch fails, fallback below
            pass

        # fallback: request one by one with retries
        embeddings: List[List[float]] = []
        for text in inputs:
            for attempt in range(3):  # up to 3 attempts
                try:
                    resp = requests.post(
                        f"{self.ollama_url}/embeddings",
                        json={"model": self.embed_model, "prompt": text},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    d = resp.json()
                    if isinstance(d, dict) and "embedding" in d:
                        embeddings.append(d["embedding"])
                        break
                    else:
                        raise RuntimeError("Unexpected response from embeddings server")
                except Exception:
                    # exponential backoff
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (attempt + 1))
        return embeddings

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Takes a list of texts and returns a numpy array with L2-normalized embeddings.
        Returns shape (n_texts, dim_embedding).
        """
        if not texts:
            return np.array([], dtype=np.float32)

        raw = self._request_embeddings(texts)
        arr = np.array(raw, dtype=np.float32)

        # normalize to avoid similarity issues
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr

    # Chat / generation
    def chat(self, prompt: str, context: str = "") -> str:
        """
        Generates a response using the chat model, injecting context if available.
        Returns the string response (expected 'response' field from Ollama).
        """
        if context:
            system = (
                "You are a technical assistant specialized in the history of computing, "
                "concise and direct. Respond based on the CONTEXT and your own knowledge.\n"
                f"Context:\n{context}"
            )
        else:
            system = "You are a concise and direct technical assistant."

        payload = {"model": self.chat_model, "prompt": prompt, "system": system, "stream": False}
        resp = requests.post(f"{self.ollama_url}/generate", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "")

    # Text fragmentation
    def fragment_text(self, text: str, chunk_size: int = 300, overlap: int = 100) -> List[Dict]:
        """
        Splits text into chunks close to 'chunk_size' characters,
        preserving sentences and applying 'overlap' between chunks.
        Each chunk is a dict: {id, text, char_start, char_end}.
        """
        # split by end punctuation (., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[Dict] = []
        current = ""
        cur_start = 0
        idx = 0

        for sent in sentences:
            if not sent.strip():
                continue
            candidate = f"{current} {sent}" if current else sent
            if len(candidate) >= chunk_size:
                end = cur_start + len(candidate)
                chunks.append({
                    "id": idx,
                    "text": candidate.strip(),
                    "char_start": cur_start,
                    "char_end": end,
                })
                idx += 1
                # prepare next chunk with overlap
                overlap_text = candidate[-overlap:] if overlap and len(candidate) > overlap else ""
                current = overlap_text
                cur_start = end - len(current)
            else:
                current = candidate

        # last chunk
        if current.strip():
            end = cur_start + len(current)
            chunks.append({"id": idx, "text": current.strip(), "char_start": cur_start, "char_end": end})

        return chunks

    # Create and save index
    def create_index_from_file(self, filename: str, chunk_size: int = 300, overlap: int = 100) -> None:
        """
        Reads a text file, fragments into chunks, generates embeddings and saves:
        - chunks.json
        - embeddings.npy
        - meta.json
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self.fragment_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise RuntimeError("No chunks created from file.")

        texts = [c["text"] for c in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        emb = self.embed(texts)

        # store only chunk metadata (text and positions)
        self.chunks = [
            {"id": c["id"], "text": c["text"], "char_start": c["char_start"], "char_end": c["char_end"]}
            for c in chunks
        ]

        # save to disk
        np.save(self.embeddings_path, emb)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        meta = {
            "source_file": os.path.abspath(filename),
            "n_chunks": len(self.chunks),
            "created_at": time.time(),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.embeddings = emb
        print("Index created and saved at:")
        print(f" - {self.chunks_path}")
        print(f" - {self.embeddings_path}")

    # Load saved index
    def load_index(self) -> bool:
        """
        Loads chunks.json and embeddings.npy into memory.
        Normalizes embeddings for safety.
        Returns True if loaded successfully, False if files don't exist.
        """
        if not os.path.exists(self.chunks_path) or not os.path.exists(self.embeddings_path):
            return False

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.embeddings = np.load(self.embeddings_path)

        # normalize for safety (avoid divide by zero)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms
        return True

    # MMR (Maximal Marginal Relevance)
    def _mmr(
        self,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
        candidate_texts: List[str],
        k: int,
        lambda_param: float = 0.6,
    ) -> List[int]:
        """
        Selects k indices from candidate_embs using MMR to balance relevance and diversity.
        Returns indices relative to candidate_embs (not global chunk indices).
        """
        sims = (candidate_embs @ query_emb).flatten()  # similarities
        selected: List[int] = []
        candidate_idxs = list(range(len(sims)))
        if k <= 0:
            return []

        # start with most relevant
        selected.append(int(np.argmax(sims)))

        while len(selected) < min(k, len(sims)):
            remaining = [i for i in candidate_idxs if i not in selected]
            mmr_scores = []
            for i in remaining:
                relevance = sims[i]
                diversity = max((candidate_embs[i] @ candidate_embs[j]) for j in selected) if selected else 0
                score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((score, i))
            # choose index with highest MMR score
            next_idx = max(mmr_scores)[1]
            selected.append(next_idx)

        return selected

    # Main search
    def search(self, query: str, top_k: int = 3, fetch_k: int = 10, use_mmr: bool = True) -> List[Tuple[Dict, float]]:
        """
        Similarity search:
        - compute embedding of query
        - rank candidates by similarity
        - apply MMR if use_mmr=True
        Returns list of (chunk_meta, score).
        """
        if self.embeddings is None:
            if not self.load_index():
                raise RuntimeError("Index not loaded. Create the index first.")

        q_emb = self.embed([query])[0]
        scores = (self.embeddings @ q_emb)
        fetch_k = min(fetch_k, len(scores))
        # take best fetch_k candidates
        top_candidate_idxs = np.argsort(-scores)[:fetch_k]
        candidate_embs = self.embeddings[top_candidate_idxs]
        candidate_texts = [self.chunks[i]["text"] for i in top_candidate_idxs]

        if use_mmr and top_k < len(candidate_texts):
            selected_offsets = self._mmr(q_emb, candidate_embs, candidate_texts, top_k)
            selected_idxs = [top_candidate_idxs[i] for i in selected_offsets]
        else:
            selected_idxs = list(top_candidate_idxs[:top_k])

        results = [(self.chunks[i], float(scores[i])) for i in selected_idxs]
        results.sort(key=lambda x: -x[1])  # sort by score desc
        return results

    # Interactive terminal interfaces
    def interactive_search(self) -> None:
        """Simple interactive search loop in terminal."""
        self.clear()
        print("RAG - interactive search. Type 'exit' to quit.")
        while True:
            q = input("Search> ").strip()
            if q.lower() in ("exit", "quit", "sair"):
                break
            try:
                hits = self.search(q, top_k=3, fetch_k=12, use_mmr=True)
            except Exception as e:
                print("Error:", e)
                continue
            if not hits:
                print("No relevant results.")
                continue
            for i, (meta, score) in enumerate(hits, 1):
                chunk_id = meta.get("id", "?")
                text = meta.get("text", "")
                chars = meta.get("char_end", 0) - meta.get("char_start", 0)
                print(f"\n[{i}] id={chunk_id} score={score:.4f} chars={chars}")
                print(text[:600] + ("..." if len(text) > 600 else ""))
            print("\n---")

    def interactive_chat(self) -> None:
        """Chat loop that gathers the best chunks as context and calls the generation model."""
        self.clear()
        print("RAG - chat with context. Type 'exit' to quit.")
        while True:
            q = input("Question> ").strip()
            if q.lower() in ("exit", "quit", "sair"):
                break
            try:
                hits = self.search(q, top_k=3, fetch_k=12, use_mmr=True)
            except Exception as e:
                print("Error:", e)
                continue

            selected_ids = [h[0].get("id", "?") for h in hits]
            print(f"\nSelected chunk ids for context: {selected_ids}\n")

            context = "\n\n".join([h[0]["text"] for h in hits])
            try:
                resp = self.chat(q, context)
            except Exception as e:
                print("Chat error:", e)
                continue
            print("\nAnswer:\n")
            print(resp)
            print("\n---")


# Main program (CLI)
def main() -> None:
    rag = LocalRAG()
    while True:
        rag.clear()
        print("Local RAG - Ollama (readable version)")
        print("1) Create index from file")
        print("2) Load saved index")
        print("3) Interactive search")
        print("4) RAG chat")
        print("0) Exit")
        opt = input("Choose> ").strip()

        if opt == "1":
            fname = input("Text file> ").strip()
            try:
                rag.create_index_from_file(fname)
            except Exception as e:
                print("Failed:", e)
                input("Press Enter to continue...")
        elif opt == "2":
            ok = rag.load_index()
            print("Index loaded." if ok else "Index not found.")
            input("Press Enter to continue...")
        elif opt == "3":
            rag.interactive_search()
        elif opt == "4":
            rag.interactive_chat()
        elif opt == "0":
            break


if __name__ == "__main__":
    # quickly check if Ollama service is listening
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except requests.exceptions.RequestException:
        print("Ollama does not seem to be running. Start with: ollama serve")
        raise SystemExit(1)
    main()
