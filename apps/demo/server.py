"""Serve the local demo UI and its one read-only, precomputed data bundle."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

from contract import (
    BUNDLE_SCHEMA_VERSION,
    ContractError,
    MovieCatalogError,
    validate_bundle,
    validate_movie_catalog,
)
from shop_catalog import CatalogError, validate_catalog


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = REPO_ROOT / "results" / "demo-bundle.json"
CATALOG_PATH = REPO_ROOT / "results" / "shop-catalog.json"
MOVIE_CATALOG_PATH = REPO_ROOT / "results" / "movie-catalog.json"
STATIC_FILES = {
    "/": (APP_DIR / "index.html", "text/html; charset=utf-8"),
    "/index.html": (APP_DIR / "index.html", "text/html; charset=utf-8"),
    "/app.js": (APP_DIR / "app.js", "text/javascript; charset=utf-8"),
    "/styles.css": (APP_DIR / "styles.css", "text/css; charset=utf-8"),
}


class DemoHandler(BaseHTTPRequestHandler):
    """A deliberately small GET-only server; no path is resolved from the URL."""

    server_version = "LocalDemo/1"

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        route = urlsplit(self.path).path
        if route == "/api/bundle":
            self._serve_bundle()
            return
        if route == "/api/catalog":
            self._serve_catalog()
            return
        if route == "/api/movies":
            self._serve_movies()
            return
        static_file = STATIC_FILES.get(route)
        if static_file is None:
            self._send(404, b"Not found\n", "text/plain; charset=utf-8")
            return
        path, content_type = static_file
        try:
            body = path.read_bytes()
        except OSError:
            self._send(500, b"Demo interface asset is unavailable\n", "text/plain; charset=utf-8")
            return
        self._send(200, body, content_type)

    def do_HEAD(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send(405, b"Method not allowed\n", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send(405, b"Method not allowed\n", "text/plain; charset=utf-8")

    def do_PUT(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send(405, b"Method not allowed\n", "text/plain; charset=utf-8")

    def do_PATCH(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send(405, b"Method not allowed\n", "text/plain; charset=utf-8")

    def do_DELETE(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send(405, b"Method not allowed\n", "text/plain; charset=utf-8")

    def _serve_bundle(self) -> None:
        try:
            parsed = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # A fresh clone has no generated results because results/ is gitignored.
            # Treat that state as an empty viewer; malformed bundles still error below.
            self._send_json(200, {"schema_version": BUNDLE_SCHEMA_VERSION, "scenarios": []})
            return
        except json.JSONDecodeError as error:
            self._send_json(422, {"error": f"Demo bundle is invalid JSON at line {error.lineno}, column {error.colno}."})
            return
        except OSError:
            self._send_json(500, {"error": "Demo bundle could not be read."})
            return
        try:
            bundle = validate_bundle(parsed)
        except ContractError as error:
            self._send_json(422, {"error": f"Demo bundle does not match the data contract: {error}"})
            return
        self._send_json(200, bundle)

    def _serve_catalog(self) -> None:
        try:
            parsed = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._send_json(200, {"schema_version": 1, "products": []})
            return
        except json.JSONDecodeError as error:
            self._send_json(422, {"error": f"Shop catalog is invalid JSON at line {error.lineno}, column {error.colno}."})
            return
        except OSError:
            self._send_json(500, {"error": "Shop catalog could not be read."})
            return
        try:
            catalog = validate_catalog(parsed)
        except CatalogError as error:
            self._send_json(422, {"error": f"Shop catalog does not match the data contract: {error}"})
            return
        self._send_json(200, catalog)

    def _serve_movies(self) -> None:
        try:
            parsed = json.loads(MOVIE_CATALOG_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._send_json(200, {"schema_version": 1, "movies": []})
            return
        except json.JSONDecodeError as error:
            self._send_json(422, {"error": f"Movie catalog is invalid JSON at line {error.lineno}, column {error.colno}."})
            return
        except OSError:
            self._send_json(500, {"error": "Movie catalog could not be read."})
            return
        try:
            catalog = validate_movie_catalog(parsed)
        except MovieCatalogError as error:
            self._send_json(422, {"error": f"Movie catalog does not match the data contract: {error}"})
            return
        self._send_json(200, catalog)

    def _send_json(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.log_date_time_string()} {format % args}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="local HTTP port (default: 8000)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    server = HTTPServer(("127.0.0.1", args.port), DemoHandler)
    print(f"Read-only demo listening at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo server stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
