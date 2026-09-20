import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health_router, chat_router, ingestion_router, evaluate_router
from app.retrieval.ingestion import run_ingestion
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Kestrel Multi-Agent Research Assistant Backend...")
    # Schedule background ingestion so server binds port immediately
    asyncio.create_task(asyncio.to_thread(run_ingestion))
    yield
    logger.info("Shutting down Kestrel Assistant Backend.")

app = FastAPI(
    title="Kestrel Labs Multi-Agent Research Assistant API",
    description="Corpus-grounded 4-Agent Research Assistant API supporting Groq, Gemini, Chroma, and LangSmith.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(ingestion_router, prefix="/api", tags=["Ingestion"])
app.include_router(evaluate_router, prefix="/api", tags=["Evaluation"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
