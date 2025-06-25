"""Module for prompt templates."""

import os

from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from lmf.example_selector import SemanticSimilarityExampleSelectorScore
from lmf.utils import make_custom_parameter

### classes for prompt templates go here ###
# these must be PromptTemplateType subclasses


class PromptTemplateType:
    """Base class for project templates."""

    prompt_template: FewShotChatMessagePromptTemplate
    prompt_args = dict(
        input_variables=["input"],
        example_prompt=ChatPromptTemplate([("human", "{input}"), ("ai", "{output}")]),
    )

    def __init__(self, **kwargs):
        pass

    def get(self) -> None | FewShotChatMessagePromptTemplate:
        """Returns initialized prompt templates."""
        return self.prompt_template


class NoTemplate(PromptTemplateType):
    """No prompt template."""

    prompt_template = None


class RandomFewShot(PromptTemplateType):
    """Random few shot prompt template (randomization implemented in main.py)."""

    prompt_template = None


class SemanticFewShot(PromptTemplateType):
    """Dynamic, semantic-similarity-based, few-shot prompt template."""

    example_selector = SemanticSimilarityExampleSelector

    def __init__(
        self,
        vector_store,
        k=4,
        semantic_example_selector_kwargs={},
        vectorstore_kwargs={"score_threshold": 0.0},
        **kwargs,
    ):
        semantic_example_selector_kwargs["k"] = k
        example_selector = self.example_selector(
            vectorstore=vector_store,
            vectorstore_kwargs=vectorstore_kwargs,
            **semantic_example_selector_kwargs,
        )
        prompt_args = self.prompt_args | kwargs
        self.prompt_template = FewShotChatMessagePromptTemplate(
            example_selector=example_selector,
            **prompt_args,
        )


class SemanticFewShotScore(SemanticFewShot):
    """Dynamic semantic similarity few-shot prompt template with similarity score."""

    example_selector = SemanticSimilarityExampleSelectorScore


class MmrFewShot(SemanticFewShot):
    """Dynamic maximal marginal relevance similarity few-shot prompt template."""

    example_selector = MaxMarginalRelevanceExampleSelector


### add new classes above this line ###
name = os.path.basename(__file__)
types = {
    k: v
    for k, v in locals().items()
    if getattr(v, "__base__", None) in [PromptTemplateType, SemanticFewShot]
}
PARAMETER = make_custom_parameter(name, types)
