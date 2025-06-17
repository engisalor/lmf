"""Module for prompt template recipes."""

import os

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from clilm.utils import make_custom_parameter

### classes for vector stores go here ###
# these must be VectorStore subclasses
# methods from the base class may need to be replaced
# when using vector stores other than InMemoryVectorStore


class PromptTemplateType:
    """Base class for project template recipes."""

    prompt_template: None | FewShotChatMessagePromptTemplate

    def __init__(self, **kwargs):
        pass

    def get(self) -> None | FewShotChatMessagePromptTemplate:
        """Returns initialized prompt templates."""
        return self.prompt_template


class NoTemplate(PromptTemplateType):
    """Recipe for no prompt template."""

    prompt_template = None


class DynamicSemanticFewShot(PromptTemplateType):
    """Recipe for dynamic, semantic-similarity-based, few-shot prompt template."""

    def __init__(
        self,
        vector_store,
        k=2,
        semantic_example_selector_kwargs={},
        **kwargs,
    ):
        semantic_example_selector_kwargs["k"] = k
        self.example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vector_store,
            **semantic_example_selector_kwargs,
        )
        prompt_args = dict(
            input_variables=["input"],
            example_prompt=ChatPromptTemplate(
                [("human", "{input}"), ("ai", "{output}")]
            ),
        )
        prompt_args |= kwargs
        self.prompt_template = FewShotChatMessagePromptTemplate(
            example_selector=self.example_selector,
            **prompt_args,
        )


### add new classes above this line ###
name = os.path.basename(__file__)
types = {
    k: v
    for k, v in locals().items()
    if getattr(v, "__base__", None) == PromptTemplateType
}
PARAMETER = make_custom_parameter(name, types)
