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

# Global converter — loaded once on startup
_converter = None


def get_converter():
    global _converter
    if _converter is None:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        print("Loading Marker models...", flush=True)
        _converter = PdfConverter(artifact_dict=create_model_dict())
        print("Models loaded.", flush=True)
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
        from marker.output import text_from_rendered

        converter = get_converter()
        rendered = converter(pdf_path)
        markdown_text, _, images = text_from_rendered(rendered)

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
