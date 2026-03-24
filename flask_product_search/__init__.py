"""Flask product search integration (loads tuned retrieval from the notebook pipeline)."""

from flask_product_search.retrieval_engine import (
    ProductRetrievalEngine,
    get_engine,
    init_engine,
)

__all__ = ["ProductRetrievalEngine", "get_engine", "init_engine"]
