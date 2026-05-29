import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from app.store.base import BaseStore
from app.store.vectordb import VectorDB

_COLLECTION_NAME = "product_features"
_BATCH_SIZE = 512


class FeatureVectorDB(BaseStore):
    def __init__(self, collection_name: str = _COLLECTION_NAME) -> None:
        self._db = VectorDB(collection_name=collection_name)

    def build_from_df(self, df: pd.DataFrame) -> None:
        le = LabelEncoder()
        brand_enc = np.array(
            le.fit_transform(df["brand"].fillna("Unknown").astype(str)),
            dtype=np.float32,
        )

        prices = pd.to_numeric(df["sales_price"], errors="coerce")
        prices = prices.fillna(prices.median() if prices.notna().any() else 0.0)
        price_norm = (
            MinMaxScaler()
            .fit_transform(np.array(prices).reshape(-1, 1))
            .flatten()
            .astype(np.float32)
        )

        cat_dicts = [v if isinstance(v, dict) else {} for v in df["parent___child_category__all"]]
        all_keys = sorted({k for d in cat_dicts for k in d})
        key_index = {k: i for i, k in enumerate(all_keys)}
        cat_matrix = np.zeros((len(df), len(all_keys)), dtype=np.float32)
        for row, d in enumerate(cat_dicts):
            for k in d:
                if k in key_index:
                    cat_matrix[row, key_index[k]] = 1.0

        matrix = np.hstack([
            brand_enc.reshape(-1, 1),
            price_norm.reshape(-1, 1),
            cat_matrix,
        ]).astype(np.float32)

        ids = df["uniq_id"].astype(str).tolist()
        metadatas = [
            {
                "brand": str(row.get("brand", "Unknown")),
                "sales_price": str(row.get("sales_price", "")),
            }
            for _, row in df.iterrows()
        ]

        total = len(ids)
        print(f"[FeatureVectorDB] Upserting {total} feature vectors ({matrix.shape[1]}d) to '{self._db.collection_name}' …")

        for start in range(0, total, _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, total)
            self._db.upsert(
                ids=ids[start:end],
                embeddings=matrix[start:end].tolist(),
                metadatas=metadatas[start:end],
            )
            if (start // _BATCH_SIZE) % 10 == 0:
                print(f"  … {end}/{total}")

        print(f"[FeatureVectorDB] Done – {total} products indexed.")

    def search(self, product_id: str, top_k: int = 2000) -> list[str]:
        embedding = self._db.get_embedding(product_id)
        if embedding is None:
            raise KeyError(f"Product '{product_id}' not found in FeatureVectorDB.")

        n_results = min(top_k + 1, self._db.count())
        returned_ids = self._db.query(embedding, n_results=n_results)
        filtered = [rid for rid in returned_ids if rid != product_id]
        return filtered[:top_k]

    def count(self) -> int:
        return self._db.count()
