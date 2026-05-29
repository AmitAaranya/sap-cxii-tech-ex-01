from typing import Any

import pandas as pd
from tqdm import tqdm

from app.store.base import BaseStore
from app.store.embedder import Embedder
from app.store.vectordb import VectorDB

_COLLECTION_NAME = "products"
_BATCH_SIZE = 256


class SemanticStore(BaseStore):
    """Store that builds and searches text-embedding vectors via ChromaDB.

    Responsibilities are split:
    - ``Embedder``  – SentenceTransformer model loading & encoding
    - ``VectorDB``  – ChromaDB collection management & querying
    - ``SemanticStore`` – DataFrame parsing and orchestration (BaseStore interface)
    """

    def __init__(
        self,
        collection_name: str = _COLLECTION_NAME,
        model_name: str | None = None,
    ) -> None:
        self._embedder = Embedder(model_name=model_name) if model_name else Embedder()
        self._db = VectorDB(collection_name=collection_name)

    def build_from_df(self, df: pd.DataFrame) -> None:
        ids: list[str] = df["uniq_id"].astype(str).tolist()
        texts: list[str] = df["text_blob"].fillna("").astype(str).tolist()
        # Vectorized metadata build — avoids slow row-by-row iteration
        product_names = df["product_name"].fillna("").astype(str).str[:512].tolist()
        brands = df["brand"].fillna("Unknown").astype(str).tolist()
        metadatas: list[dict[str, Any]] = [
            {"product_name": pn, "brand": br}
            for pn, br in zip(product_names, brands)
        ]

        total = len(ids)
        n_batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE
        print(f"[SemanticStore] Embedding {total} products with '{self._embedder.model_name}' …")

        with tqdm(total=total, unit="product", desc="Indexing") as pbar:
            for start in range(0, total, _BATCH_SIZE):
                end = min(start + _BATCH_SIZE, total)
                batch_embeddings = self._embedder.encode(texts[start:end])
                self._db.upsert(
                    ids=ids[start:end],
                    embeddings=batch_embeddings,
                    metadatas=metadatas[start:end],
                    documents=texts[start:end],
                )
                pbar.update(end - start)

        print(f"[SemanticStore] Done – {total} products indexed.")

    def search(self, product_id: str, top_k: int = 2000) -> list[str]:
        embedding = self._db.get_embedding(product_id)
        if embedding is None:
            raise KeyError(f"Product '{product_id}' not found in SemanticStore.")

        n_results = min(top_k + 1, self._db.count())
        returned_ids = self._db.query(embedding, n_results=n_results)
        filtered = [rid for rid in returned_ids if rid != product_id]
        return filtered[:top_k]

    def count(self) -> int:
        return self._db.count()
