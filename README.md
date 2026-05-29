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

## Scaling to 50 Enterprise Customers

The current service is a correct single-tenant PoC. The following changes are required before it can serve enterprise customers at scale. Each change is independently shippable; they are ordered from lowest to highest effort.

### Target Architecture

```
                         ┌──────────────────────────────────────────────────────────────────┐
                         │  Enterprise Customer (× 50)                                      │
                         │  ERP / Commerce Cloud / SAP S/4HANA                              │
                         └────────────────────────┬─────────────────────────────────────────┘
                                                  │ product upsert events
                                                  ▼
                         ┌──────────────────────────────────────────────────────────────────┐
                         │  SAP Event Mesh / Kafka                                          │
                         │  (one topic per tenant)                                          │
                         └──────┬───────────────────────────────────────────────────────────┘
                                │
                 ┌──────────────┘
                 ▼
  ┌──────────────────────────┐        ┌──────────────────────────────┐
  │   Indexing Workers       │        │   Object Store               │
  │   (auto-scaled pods)     │──────► │   S3 / GCS / SAP BTP         │
  │                          │        │   (raw catalogue per tenant) │
  │  1. fetch text_blob      │        └──────────────────────────────┘
  │  2. call Embedding Svc   │
  │  3. upsert → vector DB   │
  └──────────┬───────────────┘
             │ upsert embeddings
             ▼
┌────────────────────────────────────────────────────────────────────┐
│  Vector Store Cluster                                              │
│  ChromaDB (Stage 1) → SAP HANA Cloud Vector Engine (Stage 2)      │
│                                                                    │
│  collection: features_{tenant_id}   collection: semantic_{tenant_id} │
└────────────────────────────────────────────────────────────────────┘
             ▲                                        ▲
             │ read-only queries                      │
             └────────────────┬───────────────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │   Query Service          │
                 │   FastAPI  --workers 4   │
                 │                          │
                 │  tenant_id from JWT  ──► │──► Tenant Router
                 │  RRF fusion              │      (collection selector)
                 └────────────┬─────────────┘
                              │ encode(text_blob)
                              ▼
                 ┌────────────────────────────┐
                 │   Embedding Service        │
                 │   FastAPI + ONNX / Triton  │
                 │                            │
                 │   Redis cache              │
                 │   key: sha256(text)        │
                 └────────────────────────────┘
                              ▲
                              │ (same service used by Indexing Workers)
                              │
         ┌────────────────────┴──────────────────────────┐
         │                                               │
         │   API Gateway (mTLS)                          │
         │   X-API-Key → tenant_id resolution            │
         │   Rate limiting per tenant                    │
         └───────────────────────────────────────────────┘
                              ▲
                              │ HTTPS
                 ┌────────────┴────────────┐
                 │   Client applications   │
                 │   (per enterprise)      │
                 └─────────────────────────┘

Observability plane (cross-cutting)
─────────────────────────────────────────────────────────────────────
  Prometheus ◄── /metrics on every service (labelled by tenant_id)
  Grafana    ◄── dashboards: query p99, index staleness, cache hit rate
  Alertmanager── p99 > 200 ms │ index staleness > 15 min │ cache hit < 60%
```

---

### 1 — Remove data from the container image

**Problem:** The dataset is `COPY`-ed into the image at build time. This couples every catalogue update to a full image rebuild and push, inflates image size, and makes multi-tenant isolation impossible.

**Change:** Make the container stateless. Remove `COPY data/ data/` from the `Dockerfile`. At startup, pull the catalogue from an object store (S3, GCS, or SAP BTP Object Store) using the `CATALOGUE_URI` environment variable. Mount persistent stores via a Kubernetes `PersistentVolumeClaim` rather than relying on the container filesystem.

```dockerfile
# Remove this line from the Dockerfile
# COPY data/ data/

# Add at runtime via env var + init script
ENV CATALOGUE_URI=""
ENV CHROMA_DB_PATH=/app/stores/chroma_db
```

---

### 2 — Replace ChromaDB `PersistentClient` with a server-mode backend

**Problem:** `chromadb.PersistentClient` uses a file-backed SQLite store. It does not support concurrent access from multiple processes or hosts. The `--workers 1` caveat in the `CMD` is a direct consequence.

**Change (Stage 1 — 1–10 tenants):** Deploy ChromaDB in HTTP server mode (one pod), and point the client at it:

```python
# app/store/vectordb.py
self._client = chromadb.HttpClient(host=os.environ["CHROMA_HOST"], port=8001)
```

This immediately unlocks `--workers 4` on the query service and removes the single-process constraint.

**Change (Stage 2 — 10–50 tenants):** Migrate to a managed or clustered vector store — ChromaDB distributed, Qdrant, Weaviate, or SAP HANA Cloud Vector Engine. HANA Cloud is the preferred choice if the system-of-record is already SAP, as it eliminates a network hop and simplifies data-governance compliance.

---

### 3 — Add a tenant routing layer

**Problem:** There is no concept of a tenant. All products share a single collection, which leaks rank signals across customer catalogues and makes GDPR/data-residency compliance unenforceable.

**Change:** Introduce a `tenant_id` (resolved from the API key / JWT at the gateway) and map it to a dedicated ChromaDB collection name. The `VectorDB` class already accepts `collection_name` as a constructor argument — this requires only a routing shim on top:

```python
# app/store/vectordb.py — no change needed
VectorDB(collection_name=f"features_{tenant_id}")
VectorDB(collection_name=f"semantic_{tenant_id}")
```

Add a middleware in `main.py` that extracts `tenant_id` from the `X-API-Key` header and injects it into `request.state`. The route handler reads it and passes it to `find_similar_products`.

---

### 4 — Decouple index building from the query path (event-driven ingestion)

**Problem:** Indexes are built synchronously at cold-start from a static file. A 1 M-product catalogue takes minutes to index, there is no mechanism to update the index as catalogues change, and the query service is unavailable until the build completes.

**Change:** Extract indexing into a separate **Indexing Worker** service that consumes product upsert events from a message broker (Kafka or SAP Event Mesh, one topic per tenant). The query service is read-only and starts instantly because it connects to an already-indexed collection.

```
ERP / Commerce Cloud → Kafka topic (per tenant) → Indexing Worker → ChromaDB / Redis
                                                                           ▲
                                                          Query Service (read-only)
```

The readiness probe on the query service should verify that the target collection exists and has `count > 0` before the pod accepts traffic.

---

### 5 — Extract the embedding model as a dedicated service

**Problem:** `SentenceTransformer` (`all-MiniLM-L6-v2`) is loaded in-process in every query pod. Every pod holds a copy of the model weights in RAM. Under concurrent load the model becomes a CPU bottleneck, and changing models requires redeploying the entire query service.

**Change:** Deploy a standalone **Embedding Service** (FastAPI + ONNX Runtime or Triton Inference Server). Both the query service and the indexing workers call it via gRPC. Add a Redis cache keyed on `sha256(text)` in front of the model — repeat lookups (the same product queried by many users) never reach the model.

```python
# Replace in-process call with HTTP/gRPC
embedding = embedding_client.encode(text_blob)  # cached by content hash
```

This allows GPU nodes to be targeted exclusively for the embedding service and CPU nodes for query fanout.

---

### 6 — Kubernetes operational hardening

| Item | Current state | Required change |
|---|---|---|
| Readiness probe | None | Gate on successful collection connection + `count > 0` |
| HPA | None | CPU-based HPA for query pods; KEDA consumer-lag HPA for indexing workers |
| Workers | Forced to `1` | Remove constraint once stores are external; use `--workers 4` |
| Image size | Large (data baked in) | Stateless image; data from object store at runtime |
| Secrets | Env vars in pod spec | Move to Kubernetes `Secret` or an external secrets manager |

---

### 7 — Per-tenant observability

**Problem:** There are no metrics. At 50 enterprise customers, aggregate latency hides tenant-specific degradation.

**Change:** Instrument the following with Prometheus labels including `tenant_id`:

- Query latency p50/p99 — alert threshold: p99 > 200 ms
- Index staleness (time since last successful upsert per collection) — alert threshold: > 15 min for active ingestion pipelines
- RRF score distribution mean — sudden drops indicate embedding misalignment between the two stores
- Embedding cache hit rate — if < 60%, review TTL and key strategy

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
