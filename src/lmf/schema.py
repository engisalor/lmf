"""Module for designing structured outputs."""

import os
import re
from datetime import datetime
from typing import List, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field

from lmf.utils import camel_case, make_custom_parameter

### classes for output structures go here ###
# clear naming and descriptions improve LLM output quality


def triples_to_annotation(triples: List[dict], simplify_labels=True):
    """Converts EntityRelation.triples to JSONL; expects [{"triples": List[dict]}]"""
    annotation = dict(
        id=None,
        text="",
        entities=[],
        relations=[],
    )
    df = pd.DataFrame(triples)
    df = df.explode("triples")
    df = pd.json_normalize(df["triples"])
    # add text to annotation
    assert len(df["from_span.text"].unique()) == 1
    assert df["from_span.text"].unique()[0] == df["to_span.text"].unique()[0]
    annotation["text"] = df["from_span.text"].unique()[0]
    # add text id to annotation
    assert len(df["from_span.text_id"].unique()) == 1
    assert df["from_span.text_id"].unique()[0] == df["to_span.text_id"].unique()[0]
    annotation["id"] = int(df["from_span.text_id"].unique()[0])
    # add entities to annotation
    from_offsets = pd.json_normalize(
        df["from_span.offsets"].apply(
            lambda x: {"from_span.start_offset": x[0], "from_span.end_offset": x[1]}
        )
    )
    to_offsets = pd.json_normalize(
        df["to_span.offsets"].apply(
            lambda x: {"to_span.start_offset": x[0], "to_span.end_offset": x[1]}
        )
    )
    df = pd.merge(df, from_offsets, left_index=True, right_index=True)
    df = pd.merge(df, to_offsets, left_index=True, right_index=True)
    df = df.drop(
        [
            "from_span.offsets",
            "to_span.offsets",
            "from_span.text",
            "to_span.text",
            "from_span.text_id",
            "to_span.text_id",
        ],
        axis=1,
    )
    # NOTE deprecate this if text_before is no longer utilized
    if "from_span.text_before" in df.columns:
        df = df.drop(["from_span.text_before", "to_span.text_before"], axis=1)
    from_spans = df[[x for x in df.columns if x.startswith("from_span")]]
    to_spans = df[[x for x in df.columns if x.startswith("to_span")]]
    from_spans.columns = ["span", "start_offset", "end_offset"]
    to_spans.columns = ["span", "start_offset", "end_offset"]
    entities = pd.concat([from_spans, to_spans]).drop_duplicates()
    entities["id"] = np.random.randint(200000, 300000, len(entities))
    entities["label"] = entities["span"].apply(camel_case)
    if simplify_labels:
        entities.loc[entities["label"] != "forcedDisplacement", "label"] = "span"
    annotation["entities"] = entities[
        ["id", "label", "start_offset", "end_offset"]
    ].to_dict(orient="records")
    # merge entity ids with original data
    tmp1 = entities.rename(
        {
            "start_offset": "from_span.start_offset",
            "end_offset": "from_span.end_offset",
            "id": "from_id",
        },
        axis=1,
    )
    df = pd.merge(df, tmp1, on=["from_span.start_offset", "from_span.end_offset"])
    tmp2 = entities.rename(
        {
            "start_offset": "to_span.start_offset",
            "end_offset": "to_span.end_offset",
            "id": "to_id",
        },
        axis=1,
    )
    df = pd.merge(df, tmp2, on=["to_span.start_offset", "to_span.end_offset"])
    # add relations to annotation
    relations = df[["from_id", "to_id", "type"]].copy()
    relations["id"] = np.random.randint(300000, 400000, len(relations))
    annotation["relations"] = relations[["id", "from_id", "to_id", "type"]].to_dict(
        orient="records"
    )
    return annotation


class Unstructured(BaseModel):
    """Unstructured chat output."""


class Span(BaseModel):
    """An entity annotation containing the span being annotated and related information."""

    text_id: int = Field(description="A unique integer provided with an input text.")
    span: str = Field(description="The span being labeled as an entity.")

    @computed_field
    def text(self) -> str:
        gold = pd.read_json("projects/anno/gold.jsonl", orient="records", lines=True)
        gold = gold.set_index("id")
        return gold.loc[int(self.text_id)]["text"]

    @computed_field
    def offsets(self) -> tuple[int, int]:
        res = re.finditer(self.span.strip(), self.text)
        matches = [(m.start(0), m.end(0)) for m in res]
        if len(matches) != 1:
            return -2, -2
        start_offset, end_offset = matches[0]
        return start_offset, end_offset


class SemanticRelationTriple(BaseModel):
    """A unique semantic triple in the sentence, containing a from_span, relation type, and to_span."""

    from_span: Span
    type: Literal["caused_by"]
    to_span: Span

    @computed_field
    def log(self) -> None:
        # """Prints a detected triple for logging purposes"""
        print(datetime.now().isoformat(timespec="seconds"))
        print(self.from_span.span, ">>", self.type, ">>", self.to_span.span)
        print(self.from_span.offsets, ">>", self.to_span.offsets)


class EntityRelation(BaseModel):
    """Unique semantic triples in a sentence and the annotation object computed from the triples."""

    triples: List[SemanticRelationTriple]

    @computed_field
    def annotation(self) -> dict:
        dt = {"triples": [x.model_dump() for x in self.triples]}
        try:
            return triples_to_annotation([dt])
        except Exception as e:
            print(f"FAIL - {e} - {dt}")
            return {}


### add new classes above this line ###
name = os.path.basename(__file__)
types = {k: v for k, v in locals().items() if getattr(v, "__base__", None) == BaseModel}
PARAMETER = make_custom_parameter(name, types)


# TODO run agreement bad conversion annos to see if -2 -2 offsets
# is correctly interpreted as a no-match
# TODO how to prevent duplicate triples from being generated?
