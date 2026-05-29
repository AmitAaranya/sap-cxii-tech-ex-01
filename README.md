# Product Similarity Search

A FastAPI microservice that finds similar Amazon fashion products using a hybrid of structured feature matching and semantic embeddings.

---

## Setup

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Install dependencies

```bash
uv sync
```

### Dataset

Place `archive.zip` (containing the `.ldjson` dataset file) in the `data/` directory.
The loader reads the zip directly — no manual extraction needed.

### Run the server

```bash
uv run python main.py
```

On first start the server builds two persistent stores:

| Store | Location | Technology |
|---|---|---|
| Feature store | `feature_store.db` | SQLite |
| Vector store | `chroma_db/` | ChromaDB |

Subsequent starts reuse the stores and skip the build step.

### API

Once running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

| Endpoint | Method | Description |
|---|---|---|
| `/find_similar_products` | GET | Find products similar to a given `product_id` |
| `/search_by_text` | GET | Free-text semantic search |

**Example**

```
GET /find_similar_products?product_id=<uniq_id>&num_similar=10
```

---

## How Similarity Is Calculated

The system uses a **two-stage hybrid search** that merges structured feature similarity with semantic text similarity.

### Stage 1 — FeatureDB (structured features)

Each product is encoded into a float32 feature vector stored in SQLite:

| Dimension | Encoding |
|---|---|
| **Brand** | `LabelEncoder` integer, 1 dimension |
| **Price** | `MinMaxScaler` normalised to `[0, 1]`, 1 dimension |
| **Categories** | Multi-label one-hot from `parent___child_category__all`; one dimension per unique category key across the entire dataset |

Given a query product, **cosine similarity** is computed against all other products using the stored vectors. The top 2 000 candidates are returned ranked by similarity.

$$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

### Stage 2 — VectorDB (semantic embeddings)

Each product's `product_name` and `meta_keywords` are concatenated and encoded with the open-source model **`all-MiniLM-L6-v2`** (sentence-transformers). The 384-dimensional unit-norm embeddings are stored in ChromaDB using an HNSW index.

Given a query product, its stored embedding is retrieved and used to query the HNSW index. The top 2 000 semantically nearest products are returned.

### Stage 3 — Reciprocal Rank Fusion (RRF)

The two ranked lists (feature-based and semantic) are merged using **Reciprocal Rank Fusion**:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $k = 60$ (standard smoothing constant) and $R$ is the set of ranked lists. Every product that appears in either list receives a score; products appearing in both lists are rewarded. The merged list is sorted by RRF score descending and the top-N results are returned.

### Why this approach?

- **FeatureDB** captures structured similarity — products with the same brand, price range, and category hierarchy score highly.
- **VectorDB** captures semantic similarity — products described with similar language (e.g. "floral summer dress") score highly even across brands.
- **RRF** combines both signals without needing to tune per-score weights; items that rank well in both stores bubble to the top naturally.
