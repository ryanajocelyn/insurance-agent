"""
ChromaDB Vector Store Repository Implementation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types
from src.config import config
from src.db.vector_base import BaseVectorStoreRepository


class GoogleGenAIEmbeddingFunction:
    """Custom ChromaDB Embedding Function wrapping Google Gemini embedding models."""

    def name(self) -> str:
        return "GoogleGenAIEmbeddingFunction"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key if api_key else config.GOOGLE_API_KEY
        self.model_name = model_name if model_name else config.EMBEDDING_MODEL_NAME

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        if not self.client:
            return [[0.0] * 768 for _ in input]

        model_target = self.model_name.replace("models/", "")
        embeddings: List[List[float]] = []
        for text in input:
            try:
                res = self.client.models.embed_content(
                    model=model_target,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
                if hasattr(res, "embeddings") and res.embeddings:
                    embeddings.append(res.embeddings[0].values)
                elif hasattr(res, "embedding") and hasattr(res.embedding, "values"):
                    embeddings.append(res.embedding.values)
                elif isinstance(res, dict) and "embedding" in res:
                    embeddings.append(res["embedding"])
                elif isinstance(res, dict) and "embeddings" in res:
                    embeddings.append(res["embeddings"][0]["values"])
                else:
                    embeddings.append([0.0] * 768)
            except Exception as exc:
                print(f"[EMBEDDING WARNING] Failed to generate embedding: {exc}")
                embeddings.append([0.0] * 768)

        return embeddings

    def embed_query(self, input: Any) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        return self.embed_documents(input)


class ChromaRepository(BaseVectorStoreRepository):
    """ChromaDB implementation of the BaseVectorStoreRepository interface."""

    def __init__(self, chroma_path: Optional[Path] = None):
        self.chroma_path = chroma_path if chroma_path else config.CHROMA_DB_PATH
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )
        self.embedding_fn = GoogleGenAIEmbeddingFunction()

    def _get_collection(self, collection_name: str = "motor_policy_clauses"):
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        collection_name: str = "motor_policy_clauses",
    ) -> List[str]:
        if not documents:
            return []

        collection = self._get_collection(collection_name)

        if not ids:
            ids = [f"doc_{collection_name}_{i}" for i in range(len(documents))]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        return ids

    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        collection_name: str = "motor_policy_clauses",
    ) -> List[Dict[str, Any]]:
        collection = self._get_collection(collection_name)

        where_clause: Optional[Dict[str, Any]] = None
        if category_filter:
            where_clause = {"category": category_filter}

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause,
        )

        output_matches: List[Dict[str, Any]] = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                output_matches.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "similarity_score": round(1.0 - dist, 4) if dist <= 1.0 else 0.0,
                    }
                )

        return output_matches
