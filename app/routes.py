from fastapi import APIRouter, HTTPException, Query

from app.model import SearchResponse
from app.similarity import find_similar_products

app_router = APIRouter(tags=["Similarity Search API"])


@app_router.get("/find_similar_products", response_model=SearchResponse)
def get_similar_products(
    product_id: str = Query(..., description="The unique product ID (uniq_id)"),
    num_similar: int = Query(10, ge=1, le=100, description="Number of similar products to return"),
) -> SearchResponse:
    try:
        return find_similar_products(product_id, num_similar)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
