"""Module for designing structured outputs."""

import os
from typing import List

from pydantic import BaseModel, Field

from clilm.utils import make_custom_parameter

### classes for output structures go here ###
# these must be BaseModel subclasses & have to_json(), __repr__() methods
# clear naming and descriptions improve LLM output quality


class Unstructured(BaseModel):
    """Recipe for unstructured chat output."""


class SemanticTriple(BaseModel):
    """Recipe for outputting semantic a triple."""

    conceptA: str = Field(description="The first concept in a semantic triple.")
    relation: str = Field(
        description="The semantic relation that binds conceptA and conceptB."
    )
    conceptB: str = Field(description="The second concept in a semantic triple.")

    def to_json(self):
        return vars(self)

    def __repr__(self):
        return str(vars(self))


class SemanticTripleList(BaseModel):
    """Recipe for outputting a list of semantic triples."""

    triples: List[SemanticTriple]

    def to_json(self):
        return {"triples": [vars(x) for x in self.triples]}

    def __repr__(self):
        return str(vars(self))


### add new classes above this line ###
name = os.path.basename(__file__)
types = {k: v for k, v in locals().items() if getattr(v, "__base__", None) == BaseModel}
PARAMETER = make_custom_parameter(name, types)
