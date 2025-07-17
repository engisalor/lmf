"""Module for designing structured outputs."""

import os
from typing import List, Literal

from pydantic import BaseModel, Field

from lmf.utils import get_logger, make_custom_parameter

logger = get_logger(__name__)


class Unstructured(BaseModel):
    """Unstructured chat output."""


class UnstructuredThink(BaseModel):
    """Unstructured chat output that enforces think=True."""


class Hypernym(BaseModel):
    hypernym: str = Field(
        description="A hypernym (more generic concept) of the entity being analyzed",
        min_length=1,
        max_length=30,
    )


class Entity(BaseModel):
    """An entity analyzed for conceptual categorization in a specialized domain"""

    hyponym: str = Field(description="The entity being analyzed")
    hypernyms: List[str] = Field(description="A list of parent concepts of the hyponym")


class EntityList(BaseModel):
    """A list of categorized entities"""

    hyponym_and_harmonized_parent_concepts: List[Entity]


class SemanticRelationTriple(BaseModel):
    """A unique semantic triple in the sentence"""

    from_span: str = Field(
        description="The first entity, exactly as it appears in the text, which is the cause of the second entity"
    )
    type: Literal["caused_by"] = Field(
        description="The type(s) of semantic relations to look for, paying careful attention to the directionality between the entities"
    )
    to_span: str = Field(
        description="The second entity, exactly as it appears in the text, which is the result of the first entity"
    )


class EntityRelationExtractor(BaseModel):
    """Extracts semantic relation triples with entities linked by relations."""

    triples: List[SemanticRelationTriple] = Field(
        description="A list of unique semantic relation triples", max_length=4
    )


### add new classes above this line ###
name = os.path.basename(__file__)
types = {k: v for k, v in locals().items() if getattr(v, "__base__", None) == BaseModel}
PARAMETER = make_custom_parameter(name, types)
