import numpy as np
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"
_BATCH_SIZE = 256


class Embedder:
    """Wraps SentenceTransformer to produce normalised float32 embeddings."""

    def __init__(self, model_name: str = _EMBED_MODEL, batch_size: int = _BATCH_SIZE) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts and return normalised embeddings as nested lists."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        return np.array(embeddings).tolist()
