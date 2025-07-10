"""Module for preparing vector store recipes."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

### classes for vector stores go here ###
# these must be VectorStore subclasses
# methods from the base class may need to be replaced
# when using vector stores other than InMemoryVectorStore
from lmf.utils import get_logger, make_custom_parameter

logger = get_logger(__name__)


class VectorStoreType:
    """Base class for vector store recipes."""

    vector_store: InMemoryVectorStore

    def add_docs(self, documents: List[Document]):
        """Adds documents to a vector store.

        Examples:
            ```py
            # from list of dict
            documents = [{"page_content": "A text", "metadata": {}}]
            # from list of Document
            documents = [Document(metadata={}, page_content='A text')]

            ```
        """
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                documents[i] = Document(**doc)
        self.vector_store.add_documents(documents=documents)

    @staticmethod
    def texts_to_metadatas(texts: list):
        """Converts texts to their metadata represtation."""
        return [{k: str(v) for k, v in x.items()} for x in texts]

    @staticmethod
    def vectorize_texts(texts: list):
        """Vectorizes texts."""
        texts = [{k: str(v) for k, v in x.items()} for x in texts]
        return [" ".join([str(x) for x in e.values()]) for e in texts]

    def add_texts(self, texts: List[dict]):
        """Adds texts to a vector store.

        Examples:
            ```
            # from list of dict
            text = [{"input": "2 🦜 2", "output": "4"}]

            ```
        """
        to_vectorize = self.vectorize_texts(texts)
        metadatas = self.texts_to_metadatas(texts)
        self.vector_store.add_texts(to_vectorize, metadatas=metadatas)

    def add(self, ls: List):
        """Detects text/documents from a list of dicts and adds to vector store."""
        s = "adding examples to vector store"
        if isinstance(ls[0], Document) or ls[0].get("page_content"):
            logger.info(f"{s} - format = Documents")
            self.add_docs(ls)
        else:
            logger.info(f"{s} - format = texts")
            self.add_texts(ls)

    def get(self) -> InMemoryVectorStore:
        """Returns initialized vector store."""
        return self.vector_store

    def show_items(self, top_n=5):
        """Shows the first N items in a vector store."""
        top_n = 10
        for index, (id, doc) in enumerate(self.get().store.items()):
            if index < top_n:
                logger.debug(f"{id}")
                logger.debug(f"{doc['text']}")
                logger.debug(f"{doc['metadata']}\n")
            else:
                break


class Memory(VectorStoreType):
    """Recipe for in-memory vector store, loading examples on initialization."""

    vector_store: InMemoryVectorStore = InMemoryVectorStore

    def __init__(self, embeddings, examples, **kwargs):
        embeddings = embeddings
        self.vector_store = self.vector_store(embeddings, **kwargs)
        self.add(examples)


### add new classes above this line ###
name = os.path.basename(__file__)
types = {
    k: v for k, v in locals().items() if getattr(v, "__base__", None) == VectorStoreType
}
PARAMETER = make_custom_parameter(name, types)
