import random

from langchain_core.documents import Document
from langchain_core.example_selectors import SemanticSimilarityExampleSelector


class RandomExampleSelector(SemanticSimilarityExampleSelector):
    """Class for random selection with a memory vector store (others untested)."""

    def _documents_to_examples(self, documents: list[Document]) -> list[dict]:
        """Gets a random sample of input:output examples from 'metadata' keys."""
        examples = [dict(e["metadata"]) for e in documents]
        examples = random.sample(examples, self.k)
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    def select_examples(self, *args, **kwargs) -> list[dict]:
        """Gets all store documents, then selects a random sample."""
        example_docs = self.vectorstore.store.values()
        return self._documents_to_examples(example_docs)


class SemanticSimilarityExampleSelectorScore(SemanticSimilarityExampleSelector):
    def _documents_to_examples(self, documents: list[Document]) -> list[dict]:
        """Appends scores to example output.

        Warning: converts example to strings if objects, requires fixing.
        """
        examples = [dict(e[0].metadata) for e in documents]
        for i in range(len(examples)):
            examples[i]["output"] += f" (similarity_score={documents[i][1]:.3})"
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Includes scores with examples and enforces minimum score threshold."""
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = self.vectorstore.similarity_search_with_score(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            **vectorstore_kwargs,
        )
        threshold = vectorstore_kwargs.get("score_threshold")
        if threshold:
            example_docs = [x for x in example_docs if x[1] >= threshold]
        return self._documents_to_examples(example_docs)
