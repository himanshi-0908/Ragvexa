import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import ALLOWED_ORIGINS
from utils.rate_limit import limiter
from utils.routing import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ragvexa")

app = FastAPI(title="Ragvexa API")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # reload=True is dev-only — it watches files and restarts on every save,
    # which you never want on a real deployment. Set ENV=production for that.
    is_dev = os.getenv("ENV", "development") != "production"
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=is_dev)
