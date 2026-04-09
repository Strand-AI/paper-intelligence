"""Marker PDF conversion server for Cloudflare Containers.

Accepts raw PDF bytes via POST /convert, runs Marker, returns markdown + images.
Models are pre-loaded on startup to avoid per-request download overhead.
"""

import base64
import io
import os
import tempfile

from flask import Flask, request, jsonify

os.environ["TORCH_DEVICE"] = "cpu"

app = Flask(__name__)

# Pre-load models at import time so they're ready before the first request.
# With gunicorn --preload, this runs once in the master process and is
# inherited by workers via fork, avoiding per-worker loading overhead.
print("Loading Marker models...", flush=True)
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered as _text_from_rendered

_converter = PdfConverter(artifact_dict=create_model_dict())
print("Models loaded.", flush=True)


def get_converter():
    return _converter


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/convert", methods=["POST"])
def convert():
    pdf_data = request.get_data()
    if not pdf_data:
        return jsonify({"error": "No PDF data in request body"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_data)
        pdf_path = f.name

    try:
        converter = get_converter()
        rendered = converter(pdf_path)
        markdown_text, _, images = _text_from_rendered(rendered)

        image_list = []
        if images:
            for name, data in images.items():
                if isinstance(data, bytes):
                    b64 = base64.b64encode(data).decode()
                else:
                    buf = io.BytesIO()
                    data.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                image_list.append({"name": name, "data": b64})

        page_count = None
        if hasattr(rendered, "metadata") and isinstance(rendered.metadata, dict):
            page_count = rendered.metadata.get("page_count")

        return jsonify({
            "markdown": markdown_text,
            "images": image_list,
            "page_count": page_count,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(pdf_path)


if __name__ == "__main__":
    # Pre-load models before accepting requests
    get_converter()
    app.run(host="0.0.0.0", port=8080)
