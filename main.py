"""Application entry point.

Run with:
    python main.py
Or for production:
    uvicorn neuralops.api.main:app --host 0.0.0.0 --port 8000
"""

import sys
import asyncio

# Windows asyncpg compatibility — must be set before any uvicorn import
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from neuralops.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "neuralops.api.main:app",
        host="0.0.0.0",
        port=settings.neuralops_port,
        reload=settings.is_development,
        log_config=None,  # Use structlog instead of uvicorn's default logging
    )
