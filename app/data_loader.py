import io
import os
import zipfile

import pandas as pd

from app.store import data_store, feature_vector_store, semantic_store

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_ZIP_PATH = os.path.join(_DATA_DIR, "archive.zip")
_RAW_PATH = os.path.join(
    _DATA_DIR,
    "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
)


def _load_raw() -> pd.DataFrame:
    if os.path.exists(_ZIP_PATH):
        print(f"[DataLoader] Loading from zip: {_ZIP_PATH}")
        with zipfile.ZipFile(_ZIP_PATH, "r") as zf:
            ldjson_names = [n for n in zf.namelist() if n.endswith(".ldjson")]
            if not ldjson_names:
                raise FileNotFoundError("No .ldjson file found inside the zip archive.")
            with zf.open(ldjson_names[0]) as fh:
                raw_bytes = fh.read()
        return pd.read_json(io.BytesIO(raw_bytes), lines=True)
    elif os.path.exists(_RAW_PATH):
        print(f"[DataLoader] Loading from raw file: {_RAW_PATH}")
        return pd.read_json(_RAW_PATH, lines=True)
    raise FileNotFoundError(
        f"Dataset not found. Expected zip at '{_ZIP_PATH}' or raw file at '{_RAW_PATH}'."
    )


def init_data(*, force_rebuild: bool = False) -> None:
    print("=" * 60)
    print("[DataLoader] Initialising data pipeline …")

    df = _load_raw()
    print(f"[DataLoader] Loaded {len(df)} rows.")

    data_store.build_from_df(df)
    df = data_store.get_df() # for datastore 

    feat_populated = (not force_rebuild) and (feature_vector_store.count() == len(df))
    if force_rebuild or not feat_populated:
        print("[DataLoader] Building FeatureVectorStore …")
        feature_vector_store.build_from_df(df)
    else:
        print(f"[DataLoader] FeatureVectorStore already populated ({feature_vector_store.count()} items).")

    sem_populated = (not force_rebuild) and (semantic_store.count() == len(df))
    if force_rebuild or not sem_populated:
        print("[DataLoader] Building SemanticStore (text embeddings → ChromaDB) …")
        semantic_store.build_from_df(df)
    else:
        print(f"[DataLoader] SemanticStore already populated ({semantic_store.count()} items).")

    print(f"[DataLoader] Ready – {len(df)} products indexed.")
    print("=" * 60)
