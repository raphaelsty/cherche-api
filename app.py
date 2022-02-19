import os
import pathlib
import pickle

from cherche import compose
from fastapi import FastAPI
from pydantic import BaseModel


class Pipeline:
    """Neural search pipeline."""

    def __init__(self) -> None:
        self.model = None

    def set(self, model: compose.Pipeline) -> "Pipeline":
        self.model = model
        return self

    def get(self) -> compose.Pipeline:
        return self.model

    def __call__(self, q: str) -> list:
        return self.model(q=q)


pipeline = Pipeline()


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


@app.get("/search/", tags=["search"])
def search(q: str):
    raise ValueError(pipeline, q)
    return pipeline(q=q)


@app.on_event("startup")
def load_model():
    """Load the model when starting the API."""
    path_model = [
        file for file in os.listdir("model") if file.endswith(".pkl") or file.endswith(".pickle")
    ]

    if not path_model:
        raise ModelNotFound(
            "Model not found, you need to dump a Cherche pipeline inside the folder model with the extension .pkl or .pickle"
        )
    else:
        path_model = os.path.join("model", path_model[0])

    with open(path_model, "rb") as input_model:
        model = pickle.load(input_model)

    return pipeline.set(model=model)
