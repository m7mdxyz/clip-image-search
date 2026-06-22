import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "image-search"
DIMENSION = 512

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)


def upsert_image(image_id: str, vector: list[float], path: str) -> None:
    index.upsert(vectors=[{"id": image_id, "values": vector, "metadata": {"path": path}}])


def search(vector: list[float], top_k: int = 12) -> list[dict]:
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {"path": match["metadata"]["path"], "score": match["score"]}
        for match in result["matches"]
    ]
