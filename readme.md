# Qdrant template in python

This project shows how to use Qdrant for vector search alongside OpenAI embeddings with docker version of qdrant.

## Installation

1. Clone this repo
2. Run the following command to automatically detect your platform and install Docker:
```
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```
3. Install requirements: `pip install -r requirements.txt`
4. Get an OpenAI API key and add it to `.env` as `OPENAI_API_KEY`
5. Start the Qdrant server:

```
docker compose up -d
```

This will run a Qdrant server on port 6333 with the provided config. 

## Usage

The main functions are:

- `insert_data` - Takes a collection name, payload list, and data list. It embeds the data with OpenAI, structures it into Qdrant points, and inserts them into the collection.

- `search_data` - Takes a search term, collection name, and result limit. It embeds the search term with OpenAI and searches the Qdrant collection, returning the top matches.

To insert data:

```python
insert_data("test", [{"data":"doc1"}, {"data":"doc2"}], ["Text for doc1", "Text for doc2"]) 
```

To search:

```python
results = search_data("search text", "test")
```

The results will contain score, vector, and payload for each match.

See the bottom of the main code sample for example usage.

## Configuration

The Qdrant server uses the provided `docker-compose.yml` file and config. Environment variables are loaded from `.env`.

The default OpenAI model is `text-embedding-ada-002` but this can be changed.

The Qdrant client connects to the server on port 6333 by default.
