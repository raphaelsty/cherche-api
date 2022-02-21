# Deploy-Cherche

`deploy-search` is dedicated to deploying our neural search pipeline using [FastAPI](https://fastapi.tiangolo.com) and
[Docker](https://docs.docker.com/get-docker/). The `deploy-search` API has two routes: `/search/`
which enables the neural search pipeline to be called and an `/upload/` route which allows to
update the pipeline and or the set of indexed documents.

## Quick Start

### Clone

The first step is to clone the repository. Then we can build the container and launch it.

```sh
git clone https://github.com/raphaelsty/deploy-cherche.git
cd deploy-cherche
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

### Elasticsearch

To use the Elasticsearch retriever, the official documentation provides information to start an
independent and dedicated process [here](https://www.elastic.co/guide/en/elasticsearch/reference/7.5/docker.html).

### Debug

You can run the api locally to debug it or to modify it using:

```sh
pip install -r requirements.txt
uvicorn app:app --reload -p 80:80 
```
