# Cherche API

`deploy-search` is dedicated to deploying a Cherche API easily.

## Quick Start

### Clone

The first step is to clone the repository. Then we can build the container and launch it.

```sh
git clone https://github.com/raphaelsty/deploy-cherche.git
cd deploy-cherche
```

### Serialize

```python
>>> from cherche import retrieve, rank, data
>>> from sentence_transformers import SentenceTransformer
>>> import pickle

>>> documents = data.load_towns()

>>> retriever = retrieve.TfIdf(key="id", on=["article", "title"], documents=documents, k=30)

>>> ranker = rank.Encoder(
...       key = "id",
...       on = ["title", "article"],
...       encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...       k = 10,
... )

>>> search = retriever + ranker + documents
>>> search.add(documents)


>>> with open("model/model.pkl", "wb") as pkl_file:
...     pickle.dump(search, pkl_file)
```

### Build

```sh
docker build -t cherche .
docker run -d --name container -p 80:80 cherche
```

### Query

```sh
curl 
```

## Elasticsearch

To deploy an API with an Elastic retriever, you will need to use a dedicated container.

```sh
docker run --name es01 --net elastic -p 9200:9200 -p 9300:9300 -it docker.elastic.co/elasticsearch/elasticsearch:8.0.0
```

```python
>>> from cherche import retrieve, rank, data
>>> from sentence_transformers import SentenceTransformer
>>> import pickle

>>> documents = data.load_towns()

>>> retriever = retrieve.Elastic(key="id", on=["article", "title"], documents=documents, k=30)

>>> ranker = rank.Encoder(
...       key = "id",
...       on = ["title", "article"],
...       encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode, # device = "cuda"
...       k = 10,
... )

>>> search = retriever + ranker + documents
>>> search.add_embeddings(documents)


>>> with open("model/model.pkl", "wb") as pkl_file:
...     pickle.dump(search, pkl_file)
```

## Utils

We might want to systematically start the API when the machine hosting the container reboots. To
do so, we need enable docker a startup and use `--restart always` option.

```sh
systemctl enable docker
```

```sh
docker build -t cherche .
docker run -d --restart always --name container -p 80:80 cherche
```

## Debug

You can run the api locally to debug it or to modify it using:

```sh
uvicorn app:app --reload -p 80:80 
```
