from contextlib import asynccontextmanager
from fastapi import FastAPI


def create_app():
    from .routes import app_router
    from .data_loader import init_data

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_data()
        yield

    app = FastAPI(
        title="Product Similarity Search",
        description="Hybrid similarity search combining feature cosine similarity, "
                    "semantic search",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(app_router)
    return app
