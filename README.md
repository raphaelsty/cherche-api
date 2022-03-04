# Cherche API

`cherche-api` is dedicated to deploying our neural search pipeline using [FastAPI](https://fastapi.tiangolo.com) and
[Docker](https://docs.docker.com/get-docker/). This API has two routes: `/search/`
which enables the neural search pipeline to be called and an `/upload/` route which allows to
update the pipeline and or the set of indexed documents.

## Quick Start

### Clone

The first step is to clone the repository. Then we can build the container and launch it.

```sh
git clone https://github.com/raphaelsty/cherche-api.git
cd cherche-api
```

### Export our neural search pipeline

To deploy Cherche we will first have to define a neural search pipeline and dump it in the `model` folder with the name `model.pkl`.

```python
>>> from cherche import retrieve, rank, data
>>> from sentence_transformers import SentenceTransformer
>>> import pickle
>>> import requests

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

### Search

Search using curl:

```sh
curl "http://127.0.0.1:80/search/?q=Bordeaux"
```

Search using python:

```python
>>> import requests
>>> import urllib
>>> q = "Bordeaux"
>>> r = requests.get(f"http://127.0.0.1:80/search/?q={urllib.parse.quote(q)}")
>>> r.json()
```

### Update

We can update the neural search pipeline via the `upload` route. The `upload` route will overwrite
the `model/model.pkl` pickle file. You can also create a new container with a new model inside
`model/model.pkl`.

```python
>>> import requests
>>> import pickle
>>> from cherche import data, retrieve

>>> documents = data.load_towns()

>>> search = retrieve.TfIdf(key="id", on=["article", "title"], documents=documents, k=30)

>>> requests.post("http://127.0.0.1:80/upload/", files={"model": pickle.dumps(search)})
```

## Utils

### Restart

We might want to systematically start the API when the computer hosting the container reboots. To
do so, we need enable docker a startup and use `--restart always` option.

```sh
systemctl enable docker
```

```sh
docker build -t cherche .
docker run -d --restart unless-stopped --name container -p 80:80 cherche
```

### FastAPI

You may need to configure the Dockerfile differently to suit your needs based on the documentation provided by [FastAPI](https://fastapi.tiangolo.com/deployment/docker/).

### Elasticsearch

Suppose we wish to implement a neural search pipeline on a large corpus and update the corpus continuously. In that case, we do recommend dividing the set of documents from the API. The Elastic Retriever allows storing documents and embeddings of a ranker via Elasticsearch. This API is scalable thanks to the services provided by Elasticsearch.

The official documentation of Elasticsearch provides information to start an independent and dedicated process using Docker [here](https://www.elastic.co/guide/en/elasticsearch/reference/7.5/docker.html).

```sh
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.0.1
docker network create elastic
docker run --name es01 --net elastic -p 9200:9200 -p 9300:9300 -it docker.elastic.co/elasticsearch/elasticsearch:8.0.1
```

Remember to write down the token and password provided by Elasticsearch when running the docker. Then, after starting the Docker process dedicated to Elasticsearch, we can connect with the python client where the password field is the password provided by Elasticsearch.

We can verify that the container running Elasticsearch is working correctly:

```sh
curl --cacert http_ca.crt -u elastic https://localhost:9200
```

```python
>>> from elasticsearch import Elasticsearch, RequestsHttpConnection
>>> from cherche import retrieve

>>> es = Elasticsearch(
...    ["127.0.0.1:9200"], 
...    connection_class = RequestsHttpConnection, 
...    http_auth=('elastic', 'password'), 
...    use_ssl = True, 
...    verify_certs = False
... )

>>> retriever = retrieve.Elastic(es=es, index="towns", key="id", on=["title", "article"])
```

We can also use the proprietary [Elastic Cloud](https://cloud.elastic.co/home) service, which will make your life easier and your wallet lighter.

To deploy the API with the Elastic retriever, we will need to modify the `_load_model()` function of the file `app.py` as it is impossible to serialize the Elasticsearch client using `pickle`. In addition, we should consider using `Docker secrets` to store sensitive information such as Elasticsearch credentials.

```python
def _load_model():
    """Load the neural search pipeline."""
    from cherche import retrieve, rank
    from sentence_transformers import SentenceTransformer
    from elasticsearch import Elasticsearch, RequestsHttpConnection
    
    es = Elasticsearch(
        ["127.0.0.1:9200"], 
        connection_class = RequestsHttpConnection, 
        http_auth=('elastic', 'password'), 
        use_ssl = True, 
        verify_certs = False
    )
        
    retriever = retrieve.Elastic(es=es, index="towns", key="id", on=["title", "article"])

    ranker = rank.Encoder(
          key = "id",
          on = ["title", "article"],
          encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
          k = 10,
    )

    model = retriever + ranker
    return pipeline.set(model=model)
```

To update our Elasticsearch index of documents, we can pre-compute the embeddings of a ranker and insert them by batch from a machine with a GPU.

```python
>>> from cherche import retrieve, rank, data
>>> from elasticsearch import Elasticsearch, RequestsHttpConnection
>>> from sentence_transformers import SentenceTransformer

>>> documents = data.load_towns()

>>> es = Elasticsearch(
...    ["127.0.0.1:9200"], 
...    connection_class = RequestsHttpConnection, 
...    http_auth=('elastic', 'password'), 
...    use_ssl = True, 
...    verify_certs = False
... )

>>> retriever = retrieve.Elastic(es=es, index="towns", key="id", on=["title", "article"])

>>> ranker = rank.Encoder(
...       key = "id",
...       on = ["title", "article"],
...       encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda").encode, # Accelerate embeddings calculation using GPU
...       k=10,
... )

# Upload new documents and embeddings on Elasticsearch
>>> retriever.add_embeddings(documents=documents, ranker=ranker)
```

### Debug

You can run the api locally to debug it or to modify it using:

```sh
pip install -r requirements.txt
uvicorn app:app --reload -p 80:80 
```
