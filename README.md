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

The system uses a **three-stage hybrid pipeline** that combines structured feature matching with semantic text similarity, then merges both signals using a rank-fusion algorithm.

```
Raw LDJSON dataset
       │
       ▼
 ┌─────────────┐
 │  DataLoader  │  Reads zip / raw file → pandas DataFrame
 └──────┬──────┘
        │  df (cleaned)
        ├────────────────────────────┐
        ▼                            ▼
 ┌─────────────────┐        ┌──────────────────┐
 │  FeatureVectorDB │        │  SemanticStore   │
 │  (ChromaDB)      │        │  (ChromaDB)      │
 └────────┬────────┘        └────────┬─────────┘
          │ top-(num_similar×5) by    │ top-(num_similar×5) by
          │ cosine similarity         │ cosine similarity
          └──────────┬───────────────┘
                     ▼
            ┌─────────────────┐
            │  RRF Fusion      │  Merges both ranked lists
            └────────┬────────┘
                     ▼
             top-N results (with RRF score)
```

---

### Step 1 — Data Cleaning (`DataStore`)

Before indexing, the raw DataFrame is cleaned:

| Field | Transformation |
|---|---|
| `sales_price` | Coerced to float; missing values filled with column median |
| `weight` | Coerced to float; outliers (`> 1e8`) set to NaN, then median-filled |
| `rating` | Coerced to float; missing filled with median |
| `brand` | Null → `"Unknown"` |
| `parent___child_category__all` | Non-dict values → empty `{}` |
| `text_blob` | Concatenation of `product_name + meta_keywords + other_items_customers_buy` |

---

### Step 2 — FeatureVectorDB (structured features)

Each product is encoded into a float32 vector and stored in ChromaDB:

| Dimension(s) | Source field | Encoding |
|---|---|---|
| 1 | `brand` | `LabelEncoder` → integer |
| 1 | `sales_price` | `MinMaxScaler` → `[0, 1]` |
| N | `parent___child_category__all` | Multi-label one-hot; one dimension per unique category key across the whole dataset |

The final vector has shape `(2 + N_categories,)`.

At query time the stored vector for the query product is fetched and **cosine similarity** is computed against all other vectors via ChromaDB's HNSW index. The top `num_similar × 5` candidates are returned — e.g. 50 candidates for a request of `num_similar=10`.

$$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

> **What this captures:** Products with the same brand, similar price range, and overlapping category hierarchy score highly — regardless of how they are described.

---

### Step 3 — SemanticStore (text embeddings)

The `text_blob` field (product name + keywords + related items) for every product is encoded using the **`all-MiniLM-L6-v2`** sentence-transformers model into a 384-dimensional unit-norm embedding, stored in ChromaDB.

Encoding is done in batches of 256 with a `tqdm` progress bar. At query time the stored embedding for the query product is used to query the HNSW index and the top `num_similar × 5` semantically nearest products are returned — giving RRF enough candidates from both stores to fuse meaningfully.

> **What this captures:** Products described with similar language (e.g. "floral summer dress", "slim fit chinos") score highly even across different brands or price points.

---

### Step 4 — Reciprocal Rank Fusion (RRF)

The two ranked lists (feature-based and semantic) are merged without needing to normalise or calibrate scores from either store. The RRF formula is:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $k = 60$ (the standard smoothing constant from the original [RRF paper by Cormack et al., 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)) and $R$ is the set of ranked lists. Products appearing in both lists are naturally rewarded. The merged list is sorted descending and the top-N results returned.

#### Understanding the RRF score

The score is **not** a cosine similarity — it is a rank-fusion weight. You can recover the original rank from any score:

$$\text{rank} = \frac{1}{\text{score}} - k$$

| Example score | Calculation | Meaning |
|---|---|---|
| `0.032787` | $\frac{2}{61}$ | Ranked **#1 in both** lists |
| `0.016393` | $\frac{1}{61}$ | Ranked **#1 in one** list, absent from the other |
| `0.015625` | $\frac{1}{64}$ | Ranked **#4 in one** list only |
| `< 0.010` | — | Present in both lists but ranked lower |
| `< 0.005` | — | Appears in one list, ranked further down |

---

### Why this approach?

| Concern | Decision | Rationale |
|---|---|---|
| Structured vs semantic gap | Two separate stores | A single embedding cannot equally weight categorical brand identity and free-text semantics |
| Score calibration | RRF instead of weighted sum | RRF requires no tuning of per-score weights and is robust to different score scales |
| Index speed | ChromaDB HNSW | Approximate nearest-neighbour search scales to 30k+ products in milliseconds |
| Cold-start latency | Persistent stores | Both ChromaDB collections are built once on first start and reused on all subsequent starts |
| Embedding throughput | Batched encoding + `tqdm` | Processes 256 texts per batch with progress visibility; `show_progress_bar=True` inside sentence-transformers for per-batch feedback |

---

## Potential Enhancements

### Image-Based Visual Similarity

Currently similarity is driven entirely by structured features and text. Products that look visually alike (same colour, pattern, silhouette) but are described differently will not score well against each other. A third ChromaDB collection could close this gap:

1. **Download product images** from the `image` URL field in the dataset (async batch downloader with retries for missing URLs).
2. **Extract visual embeddings** using a pre-trained vision model such as `CLIP` (`openai/clip-vit-base-patch32`) or `EfficientNet`. CLIP is particularly well suited because it produces embeddings in the same space as text — enabling cross-modal queries like "find products that look like this description".
3. **Store in a dedicated `ImageStore`** — a third ChromaDB collection, following the same `BaseStore` interface already used by `FeatureVectorDB` and `SemanticStore`.
4. **Add to RRF fusion** — the `_reciprocal_rank_fusion` call in `similarity.py` already accepts a variable-length list of ranked lists, so the image results slot in with zero changes to the merge logic:

   ```python
   merged = _reciprocal_rank_fusion([feat_ids, sem_ids, img_ids])
   ```

5. **Graceful fallback** — products with no image URL are simply absent from the image ranked list. RRF handles this naturally; their score is determined entirely by the other two signals.

This approach completes the multimodal pipeline described in the task brief: structured features + text semantics + visual appearance, all fused via RRF.
