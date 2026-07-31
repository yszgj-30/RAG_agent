from embeddings.ollama_embedding import OllamaEmbeddings


class FakeResponse:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def raise_for_status(self):
        return None

    def json(self):
        return {"embeddings": self._embeddings}


class FakeClient:
    def __init__(self):
        self.requests = []

    def post(self, url, json):
        self.requests.append((url, json))
        return FakeResponse([[float(len(text))] for text in json["input"]])


def test_ollama_embeddings_batch_and_query():
    client = FakeClient()
    embeddings = OllamaEmbeddings(
        client=client,
        batch_size=2,
        model="local-embedding",
    )

    assert embeddings.embed_documents(["a", "bb", "ccc"]) == [
        [1.0],
        [2.0],
        [3.0],
    ]
    assert embeddings.embed_query("query") == [5.0]
    assert len(client.requests) == 3
    assert client.requests[0][1]["model"] == "local-embedding"
