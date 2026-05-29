import numpy as np
import pandas as pd

from app.store.base import BaseStore


class DataStore(BaseStore):
    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None

    def build_from_df(self, df: pd.DataFrame) -> None:
        self._df = self._clean(df)
        print(f"[DataStore] Loaded {len(self._df)} products from external DataFrame.")

    def search(self, product_id: str, top_k: int = 2000) -> dict:
        if self._df is None:
            raise RuntimeError("Call load() first.")
        row = self._df[self._df["uniq_id"] == product_id]
        if row.empty:
            raise KeyError(f"Product '{product_id}' not found in DataStore.")
        return row.iloc[0].to_dict()

    def get_df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call load() first.")
        return self._df

    def count(self) -> int:
        return len(self._df) if self._df is not None else 0

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["price"] = pd.to_numeric(df["sales_price"], errors="coerce")
        df["weight_num"] = pd.to_numeric(df["weight"], errors="coerce")
        df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
        df.loc[df["weight_num"] > 1e8, "weight_num"] = np.nan

        for col in ("price", "weight_num", "rating_num"):
            median = df[col].median()
            df[col] = df[col].fillna(median if pd.notna(median) else 0.0)

        df["brand_clean"] = df["brand"].fillna("Unknown").astype(str)
        df["delivery"] = df["delivery_type"].fillna("unknown").astype(str)
        df["prime"] = (df["amazon_prime__y_or_n"] == "Y").astype(int)
        df["parent___child_category__all"] = df["parent___child_category__all"].apply(
            lambda v: v if isinstance(v, dict) else {}
        )
        df["text_blob"] = (
            df["product_name"].fillna("")
            + " " + df["meta_keywords"].fillna("")
            + " " + df["other_items_customers_buy"].fillna("").astype(str)
        ).str.strip()
        df["uniq_id"] = df["uniq_id"].astype(str)

        return df
