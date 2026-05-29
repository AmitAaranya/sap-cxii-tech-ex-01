from pydantic import BaseModel

class Product(BaseModel):
    id: str
    brand: str
    price_raw: float
    price_norm: float
    categories: list[str]

class SimilarProduct(BaseModel):
    id: str
    score: float

class SearchResponse(BaseModel):
    data: list[SimilarProduct]
