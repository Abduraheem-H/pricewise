"""`python -m pricewise.serve` -> launch the API with uvicorn."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"PriceWise serving on http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("pricewise.serve.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
