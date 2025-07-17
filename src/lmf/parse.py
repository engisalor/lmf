"""Classes to parse output schemas."""

import re

import numpy as np
import pandas as pd
import regex
from pydantic import BaseModel

from lmf.command import Command
from lmf.io import YamlLoader
from lmf.utils import camel_case, get_logger

logger = get_logger(__name__)

# Parsers


class Parser:
    """The base schema parser class."""

    command: Command
    responses: list[BaseModel]


class YamlParser(Parser):
    def __init__(self, command: Command, responses: list[BaseModel]):
        YamlLoader.save_yaml(responses, file=command.yaml_out)


class EntityRelationExtractorParser(Parser):
    """Parses entity relation triples and saves to JSONL annotations."""

    def fuzzy_match(self, text, span, id=None):
        r = regex.compile(f"({re.escape(span)}){{i<=6:\s}}", re.IGNORECASE)
        res = r.finditer(text)
        matches = [(m.start(0), m.end(0)) for m in res]
        if len(matches) == 1:
            start_offset, end_offset = matches[0]
            if text[start_offset:end_offset].startswith(" "):
                start_offset += 1
            if text[start_offset:end_offset].endswith(" "):
                end_offset -= 1
            msg = f"{id}"
            msg += f"\nfuzzy.old {span}"
            msg += f"\nfuzzy.new {text[start_offset:end_offset]}"
            logger.info(msg)
            return start_offset, end_offset
        else:
            msg = f"fail - {len(matches)} found - {id}"
            msg += f"\nfuzzy.span {span}\nfuzzy.text {text}"
            logger.warning(msg)
            return 0, 0

    def offsets(self, row: pd.Series, span_name) -> tuple[int, int]:
        if span_name in row.index:
            span = row[span_name].strip()
            res = re.finditer(re.escape(span), row["text"], re.IGNORECASE)
            matches = [(m.start(0), m.end(0)) for m in res]
            if len(matches) == 1:
                start_offset, end_offset = matches[0]
                return start_offset, end_offset
            else:
                return self.fuzzy_match(row["text"], span, row["id"])
        else:
            return 0, 0

    def triples_to_annotation(self, df: pd.DataFrame, simplify_labels=True):
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
        df["from_offsets"] = df.apply(self.offsets, span_name="from_span", axis=1)
        df["to_offsets"] = df.apply(self.offsets, span_name="to_span", axis=1)
        bad_match = df[(df["from_offsets"] != (0, 0)) & (df["to_offsets"] != (0, 0))]
        if len(bad_match):
            annotation["Comments"].append(f"bad_match={len(bad_match)}")
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
                entities.loc[entities["label"] != "forcedDisplacement", "label"] = (
                    "span"
                )
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
            df = pd.merge(
                df, tmp1, on=["from_span.start_offset", "from_span.end_offset"]
            )
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
            annotation["relations"] = relations[
                ["id", "from_id", "to_id", "type"]
            ].to_dict(orient="records")
            return annotation

    def __init__(self, command: Command, responses: list[BaseModel]):
        logger.info(f" {self.__name__} - converting responses to JSONL")
        gold = pd.read_json(command.gold_standard_file, orient="records", lines=True)
        jsonl_file = command.output_file.with_suffix(f".{command.run}.jsonl")
        responses = [x.model_dump() for x in responses]
        gold = gold[: command.sample]
        df = pd.DataFrame.from_records(responses)
        df["id"] = gold["id"]
        df["text"] = gold["text"]
        df["file"] = jsonl_file.name
        records = df.groupby("id")[df.columns].apply(self.triples_to_annotation)
        for record in records:
            record["Comments"] = [jsonl_file.with_suffix("").name] + record["Comments"]
        logger.info(f"saving {len(responses)} responses")
        records.to_json(
            jsonl_file,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        logger.info(f"saving gold standard with LLM annotations")
        gold.to_json(
            command.output_dir / command.gold_standard_file.name,
            orient="records",
            lines=True,
            force_ascii=False,
        )
