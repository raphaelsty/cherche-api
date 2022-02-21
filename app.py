import os
import pickle

from cherche import compose
from fastapi import FastAPI, File, HTTPException
from pydantic import BaseModel


class Pipeline:
    """Neural search pipeline."""

    def __init__(self, model=None) -> None:
        self.model = model

    def set(self, model) -> "Pipeline":
        self.model = model
        return self

    def get(self) -> compose.Pipeline:
        return self.model

    def __call__(self, q: str) -> list:
        return self.model(q=q)


class ModelNotFound(Exception):
    pass


class Model(BaseModel):
    model: compose.Pipeline

    class Config:
        arbitrary_types_allowed = True


app = FastAPI(
    openapi_tags=[
        {
            "name": "view",
            "description": "View your neural search pipeline.",
        },
        {
            "name": "search",
            "description": "Search for a document.",
        },
        {
            "name": "upload",
            "description": "Upload a neural search pipeline.",
        },
    ],
    description="Neural Search api with Docker and Cherche.",
    title="Neural Search",
    version="0.0.1",
)

pipeline = Pipeline()


def _load_model():
    """Load the neural search pipeline."""
    if not os.path.isfile("model/model.pkl"):
        return None
    try:
        with open("model/model.pkl", "rb") as input_model:
            model = pickle.load(input_model)
    except:
        return None

    return pipeline.set(model=model)


@app.get("/search/", tags=["search"])
def search(q: str):
    if pipeline.get() is None:
        _load_model()
        if pipeline.get() is None:
            raise HTTPException(status_code=503, detail="Neural search pipeline not found.")

    documents = pipeline(q=q)
    for document in documents:
        document["similarity"] = document["similarity"].astype(float)

    return documents


@app.post("/upload/", tags=["upload"])
def upload(model: bytes = File(...)):
    import os

    print(os.listdir("./"))
    with open("model/model.pkl", "wb") as f:
        f.write(model)
    _load_model()
    return {"Neural search pipeline uploaded."}


@app.on_event("startup")
def load_model():
    """Load the model when starting the API."""
    import os

    print(os.listdir("./"))
    return _load_model()
