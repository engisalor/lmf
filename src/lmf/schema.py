"""Module for designing structured outputs."""

import os
import re
from typing import List, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from lmf.utils import camel_case, make_custom_parameter

### classes for output structures go here ###
# clear naming and descriptions improve LLM output quality


class Unstructured(BaseModel):
    """Unstructured chat output."""


class SemanticRelationTriple(BaseModel):
    """A unique semantic triple in the sentence, containing a from_span, relation type, and to_span."""

    from_span: str = Field(
        description="The first entity in the triple, which is the cause of the second entity"
    )
    type: Literal["caused_by"]
    to_span: str = Field(
        description="The second entity in the triple, which is the result of the first entity"
    )


class EntityRelationExtractor(BaseModel):
    """Extracts semantic relation triples with entities linked by relations."""

    triples: List[SemanticRelationTriple] = Field(
        description="A list of unique semantic relation triples", max_length=4
    )


def offsets(row: pd.Series, span_name) -> tuple[int, int]:
    if span_name in row.index:
        res = re.finditer(re.escape(row[span_name].strip()), row["text"])
        matches = [(m.start(0), m.end(0)) for m in res]
        if len(matches) == 1:
            start_offset, end_offset = matches[0]
            return start_offset, end_offset
        else:
            return 0, 0
    else:
        return 0, 0


def triples_to_annotation(df: pd.DataFrame, simplify_labels=True):
    """Converts EntityRelation.triples to JSONL; expects [{"triples": List[dict]}]"""
    annotation = dict(
        id=None,
        text="",
        entities=[],
        relations=[],
        Comments=[],
    )
    # reshape data
    df = df.explode("triples")
    df = pd.merge(
        df.reset_index(drop=True),
        pd.json_normalize(df["triples"]),
        left_index=True,
        right_index=True,
    )
    annotation["id"] = df.iloc[0]["id"]
    annotation["text"] = df.iloc[0]["text"]
    df = df.drop("triples", axis=1)
    # get offsets
    df["from_offsets"] = df.apply(offsets, span_name="from_span", axis=1)
    df["to_offsets"] = df.apply(offsets, span_name="to_span", axis=1)
    bad_match = df[(df["from_offsets"] != (0, 0)) & (df["to_offsets"] != (0, 0))]
    if len(bad_match):
        annotation["Comments"].append(f"bad_match={len(bad_match)}")
        print(f"... WARNING dropping {len(bad_match)} bad matches")
    # drop bad matches
    df = df[(df["from_offsets"] != (0, 0)) & (df["to_offsets"] != (0, 0))]
    # get spans w/ offsets
    df["from_span.start_offset"] = df["from_offsets"].apply(lambda x: x[0])
    df["from_span.end_offset"] = df["from_offsets"].apply(lambda x: x[1])
    df["to_span.start_offset"] = df["to_offsets"].apply(lambda x: x[0])
    df["to_span.end_offset"] = df["to_offsets"].apply(lambda x: x[1])
    from_spans = df[[x for x in df.columns if x.startswith("from_span")]]
    to_spans = df[[x for x in df.columns if x.startswith("to_span")]]
    # return empty annotation
    if not "from_span" in from_spans.columns or not "to_span" in to_spans.columns:
        return annotation
    # process entities and relations
    else:
        from_spans.columns = ["span", "start_offset", "end_offset"]
        to_spans.columns = ["span", "start_offset", "end_offset"]
        entities = pd.concat([from_spans, to_spans]).drop_duplicates()
        entities["id"] = np.random.randint(1000000, 2000000, len(entities))
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
        relations["id"] = np.random.randint(2000000, 3000000, len(relations))
        annotation["relations"] = relations[["id", "from_id", "to_id", "type"]].to_dict(
            orient="records"
        )
        return annotation


### add new classes above this line ###
name = os.path.basename(__file__)
types = {k: v for k, v in locals().items() if getattr(v, "__base__", None) == BaseModel}
PARAMETER = make_custom_parameter(name, types)
