import os
from typing import Any

import chromadb
from chromadb.api import ClientAPI

_DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


class VectorDB:
    """Pure ChromaDB wrapper — handles collection management, upsert, and vector queries.

    Has no knowledge of embedding models or DataFrames.
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client: ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def _get_client(self) -> ClientAPI:
        if self._client is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Insert or update vectors in the collection."""
        kwargs: dict[str, Any] = {"ids": ids, "embeddings": embeddings}
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        if documents is not None:
            kwargs["documents"] = documents
        self._get_collection().upsert(**kwargs)

    def get_embedding(self, doc_id: str) -> list[float] | None:
        """Fetch the stored embedding for a single document id. Returns None if not found."""
        result = self._get_collection().get(ids=[doc_id], include=["embeddings"])
        if not result["ids"]:
            return None
        return result["embeddings"][0]  # type: ignore[index]

    def query(
        self,
        embedding: list[float],
        n_results: int,
    ) -> list[str]:
        """Return the top-n_results document ids closest to the given embedding."""
        result = self._get_collection().query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["distances"],
        )
        return result["ids"][0]

    def count(self) -> int:
        return self._get_collection().count()
