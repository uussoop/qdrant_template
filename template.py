from re import template
import qdrant_client
import openai
from qdrant_client.http.models import PointStruct
from qdrant_client.http import models

from dotenv import load_dotenv
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# default host if not changed inside the docker-compose file
QDRANT_HOST = "http://0.0.0.0:6333/"

qd_client = qdrant_client.QdrantClient(url=QDRANT_HOST)


def format_embedding_to_db_structure(data):
    embeddings = openai.Embedding.create(
        input=data["chat analysis with detected patterns"],
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY,
    )
    data1 = embeddings["data"]  # type: ignore

    if len(data1) > 0:
        structured = []
        for d in data1:
            structured.append(
                PointStruct(
                    id=qd_client.get_collection(collection_name="test").vectors_count,
                    vector=d["embedding"],
                    payload=data,
                )
            )
        return structured, len(data1[0]["embedding"])
    else:
        return None


def template_filler(analysis, future_pattern):
    template = {
        "chat analysis with detected patterns": "",
        "future pattern of market": "",
    }
    template["chat analysis with detected patterns"] = analysis
    template["future pattern of market"] = future_pattern
    return template


def insert_data(collection_name, data):
    points = format_embedding_to_db_structure(data)
    if collection_name not in [c.name for c in qd_client.get_collections().collections]:
        qd_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=points[1], distance=models.Distance.COSINE  # type: ignore
            ),
        )
    qd_client.upsert(collection_name=collection_name, wait=True, points=points[0])  # type: ignore
    return True


def search_data(search_term, limit=3):
    embeddings = openai.Embedding.create(
        input=search_term, model="text-embedding-ada-002", api_key=OPENAI_API_KEY
    )
    data = embeddings["data"][0]["embedding"]  # type: ignore
    search_result = qd_client.search(
        collection_name="test",
        query_vector=data,
        limit=limit,
        with_payload=True,
        with_vectors=True,
    )

    return [
        {
            "score": result.score,
            "vector": result.vector,
            "payload": result.payload,
        }
        for result in search_result
    ]


# example usage
# before using : docker compose up -d
# insert_data("test", template_filler("pattern 1", "predction"))
# search_data("a12g21222")
