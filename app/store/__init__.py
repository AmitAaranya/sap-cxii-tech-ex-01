from typing import cast

from app.store.data_store import DataStore
from app.store.feature_store import FeatureVectorDB
from app.store.semantic_store import SemanticStore

__all__ = ["data_store", "feature_vector_store", "semantic_store"]

data_store = DataStore()
feature_vector_store =  FeatureVectorDB()
semantic_store = SemanticStore()
