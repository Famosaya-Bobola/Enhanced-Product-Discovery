"""
Run from the **project root** (the folder that contains `data/` and `flask_product_search/`):

    pip install flask
    python -m flask --app flask_product_search.app run --debug

Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from flask_product_search.retrieval_engine import get_engine, init_engine

_engine_initialized = False


def create_app() -> Flask:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(pkg_dir, "templates"),
        static_folder=os.path.join(pkg_dir, "static"),
    )

    @app.before_request
    def _load_retrieval_once():
        global _engine_initialized
        if not _engine_initialized:
            init_engine()
            _engine_initialized = True

    @app.route("/")
    def index():
        return render_template("search.html")

    @app.route("/api/search")
    def api_search():
        q = (request.args.get("q") or "").strip()
        mode = (request.args.get("mode") or "hybrid").lower()
        try:
            top_k = min(int(request.args.get("top_k", 20)), 100)
        except ValueError:
            top_k = 20

        if not q:
            return jsonify({"query": q, "mode": mode, "results": []})

        eng = get_engine()
        if mode == "bm25":
            ranked = eng.search_bm25(q, top_k=top_k)
        elif mode == "dense":
            ranked = eng.search_dense(q, top_k=top_k)
        else:
            ranked = eng.search_hybrid(q, top_k=top_k)

        return jsonify(
            {
                "query": q,
                "mode": mode,
                "results": eng.results_payload(ranked),
            }
        )

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()
