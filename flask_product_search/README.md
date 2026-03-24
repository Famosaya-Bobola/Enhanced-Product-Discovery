# Flask product search (notebook → web)

This mini-app loads the **same tuned settings** as your Jupyter pipeline:

| File | Role |
|------|------|
| `benchmark/retrieval_config.json` | BM25 `(k1, b, ε)`, hybrid dense leg (SBERT vs E5), RRF `k0` / `pool_k` |
| `data/products.csv` | Catalog (preprocessing matches the notebook: `lexical_text` / `semantic_text`) |
| `artifacts/*_embeddings.npy` | Optional: skip re-encoding the corpus at startup if row count matches |

## Run locally

From the **repository root** (`omobobola-project/`), with your conda env activated:

```bash
pip install flask
python -m flask --app flask_product_search.app run --debug
```

Open **http://127.0.0.1:5000**.

- **First request** builds BM25 + loads the sentence encoder (+ encodes the corpus if `.npy` is missing or wrong size). This can take several minutes on CPU.
- **API:** `GET /api/search?q=...&mode=hybrid|bm25|dense&top_k=20`

### Environment overrides

| Variable | Default |
|----------|---------|
| `PRODUCTS_CSV` | `data/products.csv` |
| `RETRIEVAL_CONFIG` | `benchmark/retrieval_config.json` |
| `ARTIFACTS_DIR` | `artifacts` (if the directory does not exist, it is ignored) |

## Production-ish deployment

1. **Precompute embeddings** in the notebook and keep `artifacts/sbert_embeddings.npy` or `e5_embeddings.npy` aligned with the same preprocessed row count as `products.csv` (re-run the notebook after changing the catalog).
2. Run behind **gunicorn** (Linux/macOS):

   ```bash
   pip install gunicorn
   gunicorn -w 1 -b 0.0.0.0:8000 "flask_product_search.app:app"
   ```

   Use **`-w 1`** (or load the heavy model in a worker initializer) so you do not duplicate the encoder in RAM per worker.

3. Put **nginx** or **Cloud Load Balancer** in front for TLS and static caching.

## Keeping search in sync with the notebook

1. Refresh `data/products.csv` (your preparation notebook).
2. Re-run retrieval tuning + final cells so `retrieval_config.json` and optional `.npy` files update.
3. Restart the Flask process.

If preprocessing in Python ever diverges from the notebook, export a small `retrieval_corpus.parquet` from the notebook (with `doc_id`, `lexical_text`, `semantic_text`) and point the app to that file instead of re-deriving from CSV—same idea, stricter alignment.
