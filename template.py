from re import template
import qdrant_client
from qdrant_client import QdrantClient
import openai
from qdrant_client.http.models import PointStruct
from qdrant_client.http import models

from dotenv import load_dotenv
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# default host if not changed inside the docker-compose file
QDRANT_HOST = "http://0.0.0.0:6333/"


# if you would like to run without docker use below
# qd_client = QdrantClient(":memory:")
# # or
# qd_client = QdrantClient(path="path/to/db")

qd_client = qdrant_client.QdrantClient(url=QDRANT_HOST)


def format_embedding_to_db_structure(data, payload, collection_name):
    embeddings = openai.Embedding.create(
        input=data,
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY,
    )
    data1 = embeddings["data"]  # type: ignore

    if len(data1) > 0:
        structured = []

        try:
            vc = (
                qd_client.get_collection(collection_name=collection_name).vectors_count
                + 1
            )

        except:
            vc = 0
        for d in data1:
            # print("collection none existant")
            structured.append(
                PointStruct(
                    id=vc,
                    vector=d["embedding"],
                    payload=payload[d["index"]],
                )
            )
            vc += 1
        return structured, len(structured[0].vector)
    else:
        return None


def insert_data(collection_name, data, payload):
    points = format_embedding_to_db_structure(data, payload, collection_name)
    if collection_name not in [c.name for c in qd_client.get_collections().collections]:
        qd_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=points[1], distance=models.Distance.COSINE  # type: ignore
            ),
        )
    for p in points[0]:  # type: ignore
        qd_client.upsert(collection_name=collection_name, wait=True, points=[p])  # type: ignore
    return True


def search_data(search_term, collection_name, limit=3):
    embeddings = openai.Embedding.create(
        input=search_term, model="text-embedding-ada-002", api_key=OPENAI_API_KEY
    )
    data = embeddings["data"][0]["embedding"]  # type: ignore
    if collection_name not in [c.name for c in qd_client.get_collections().collections]:
        return None
    search_result = qd_client.search(
        collection_name=collection_name,
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


# if __name__ == "__main__":
# example usage
# before using : docker compose up -d
# quastion = ""
# answer = ""
# insert_data("test", [quastion], [{"data": answer}])
# s = search_data(quastion, "test")
# for i in s:  # type: ignore
#     print(i["payload"])
#     break
