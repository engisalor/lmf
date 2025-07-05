import itertools
import json
import shutil
from collections.abc import Callable
from datetime import datetime
from hashlib import blake2b
from multiprocessing import Pool
from pathlib import Path
from typing import List

import click
import pandas as pd
import sklearn


class PairwiseAgreement:
    """Class for calculating pairwise agreement."""

    def entities(self):
        """Extracts entity annotations from file data and stores in self.entity DataFrame."""
        # extract json data
        self.entity = self.source[["annotator", "text", "entities", "text_id"]].copy(
            deep=True
        )
        self.entity = self.entity.explode("entities")
        tmp = pd.json_normalize(self.entity["entities"])
        self.entity.drop("entities", axis=1, inplace=True)
        self.entity.reset_index(drop=False, inplace=True)
        # initial processing
        df = pd.merge(tmp, self.entity, left_index=True, right_index=True)
        # add span based on offsets
        df[["start_offset", "end_offset"]] = (
            df[["start_offset", "end_offset"]].fillna(-1).astype("int64")
        )
        df["span"] = df.apply(
            lambda row: row["text"][row["start_offset"] : row["end_offset"]], axis=1
        )
        df.drop("text", axis=1, inplace=True)
        # strip strings in case a span includes whitespace
        df["label"] = df["label"].fillna("")
        df[["label", "span"]] = df[["label", "span"]].apply(lambda x: x.str.strip())
        # create interval index based on offsets
        df.index = pd.IntervalIndex.from_arrays(df["start_offset"], df["end_offset"])
        self.entity = df

    def relations(self):
        """Extracts relation annotations from file data and stores in self.relation DataFrame."""
        # extract json data
        self.relation = self.source[["annotator", "relations", "text_id"]].copy(
            deep=True
        )
        self.relation = self.relation.explode("relations")
        tmp_normalized = pd.json_normalize(self.relation["relations"])
        self.relation.drop("relations", axis=1, inplace=True)
        self.relation.reset_index(drop=False, inplace=True)
        # initial processing
        df = pd.merge(tmp_normalized, self.relation, left_index=True, right_index=True)
        df.rename({"id": "rel_id"}, axis=1, inplace=True)
        df.drop_duplicates(
            ["annotator", "from_id", "to_id", "index", "type"], inplace=True
        )
        # add to and from columns
        columns_preserve = {"annotator": "annotator", "index": "index"}
        column_map_from = columns_preserve | {
            "id": "from_id",
            "start_offset": "from_start_offset",
            "end_offset": "from_end_offset",
            "span": "from_span",
        }
        column_map_to = columns_preserve | {
            "id": "to_id",
            "start_offset": "to_start_offset",
            "end_offset": "to_end_offset",
            "span": "to_span",
        }
        df[["rel_id", "from_id", "to_id"]] = (
            df[["rel_id", "from_id", "to_id"]].fillna(-1).astype("int64")
        )
        df = pd.merge(
            df,
            self.entity.rename(column_map_from, axis=1)[
                [x for x in column_map_from.values()]
            ],
            on=["from_id", "index", "annotator"],
            how="left",
        )
        df = pd.merge(
            df,
            self.entity.rename(column_map_to, axis=1)[
                [x for x in column_map_to.values()]
            ],
            on=["to_id", "index", "annotator"],
            how="left",
        )
        # strip strings in case a span includes whitespace
        df[["type", "from_span", "to_span"]] = (
            df[["type", "from_span", "to_span"]]
            .fillna("")
            .apply(lambda x: x.str.strip())
        )
        # create interval index based on offsets
        cols = [
            "from_start_offset",
            "from_end_offset",
            "to_start_offset",
            "to_end_offset",
        ]
        df[cols] = df[cols].fillna(-1).astype("int64")
        df["type"] = df["type"].fillna("")
        df.index = pd.IntervalIndex.from_arrays(
            df["from_start_offset"], df["from_end_offset"]
        )
        df.set_index(
            pd.IntervalIndex.from_arrays(df["to_start_offset"], df["to_end_offset"]),
            append=True,
            inplace=True,
        )
        # final adjustments
        df.drop(
            [
                "from_start_offset",
                "from_end_offset",
                "to_start_offset",
                "to_end_offset",
            ],
            axis=1,
            inplace=True,
        )
        self.relation = df

    @staticmethod
    def _add_solitary_annotations(
        group: pd.DataFrame, pairs: List[pd.DataFrame], id_col
    ) -> List[pd.DataFrame]:
        annotators = group["annotator"].unique().tolist()
        ls_all = pairs
        solitary = [
            x
            for x in group[id_col].unique()
            if x not in set([y for df in pairs for y in df[id_col]])
        ]
        for s in solitary:
            # _df1 is the solitary annotation
            _df1 = group[group[id_col] == s].copy()
            if not _df1.empty:
                solitary_annotator = _df1.iloc[0]["annotator"]
                # _df2 is the new row to match with the solitary annotation
                _df2 = group[group[id_col] == s].copy()
                _df2[id_col] = 0
                if id_col == "id":
                    _df2["label"] = ""
                elif id_col == "rel_id":
                    _df2["type"] = ""
                _df2["annotator"] = [x for x in annotators if x != solitary_annotator][
                    0
                ]
                _df3 = pd.concat([_df1, _df2])
                _df3.drop_duplicates(inplace=True)
                _df3.sort_values([id_col], inplace=True)
                _df3.reset_index(drop=True, inplace=True)
                _bytes = bytes(
                    json.dumps(_df3.to_dict("records"), sort_keys=True),
                    encoding="utf-8",
                )
                _df3["hash"] = blake2b(_bytes).hexdigest()[:32]
                ls_all.append(_df3)
        return ls_all

    def match(self, debug=False):
        # always process entities
        if not debug:
            with Pool() as p:
                entity_match = p.map(self._match, self.entity.groupby("index"))
                self.entity_match = pd.concat(entity_match)
        # process relation if any
        if not debug and not self.relation.empty:
            with Pool() as p:
                relation_match = p.map(self._match, self.relation.groupby("index"))
                self.relation_match = pd.concat(relation_match)
        # debug: always process entities
        if debug:
            for x, text in self.entity.groupby("index"):
                print("\nENTITY", x)
                print(text)
                self.entity_match.append(self._match(text))
            self.entity_match = pd.concat(self.entity_match)
        # debug: process relation if any
        if debug and not self.relation.empty:
            for x, text in self.relation.groupby("index"):
                print("\nRELATION", x)
                print(text)
                self.relation_match.append(self._match(text))
            self.relation_match = pd.concat(self.relation_match)
            self.relation_match.reset_index(drop=True, inplace=True)

        self.entity_match.reset_index(drop=True, inplace=True)

    def _match(self, groupby_tuple) -> pd.DataFrame:
        _, text = groupby_tuple
        text: pd.DataFrame
        pairs = []
        id_col = "id"
        if "rel_id" in text.columns:
            id_col = "rel_id"
        for x, y in itertools.combinations(text.index, 2):
            df = text.loc[[x, y]].copy(deep=True)
            df.drop_duplicates(inplace=True)
            df.sort_values([id_col], inplace=True)
            df.reset_index(drop=True, inplace=True)
            _bytes = bytes(
                json.dumps(df.to_dict("records"), sort_keys=True), encoding="utf-8"
            )
            df["hash"] = blake2b(_bytes).hexdigest()[:32]
            # full match: both indexes
            if x == y:
                pairs.append(df)
            elif not df["annotator"].duplicated().any():
                # overlapping match entities
                if id_col == "id" and x.overlaps(y):
                    pairs.append(df)
                # overlapping match relations
                elif id_col == "rel_id" and x[0].overlaps(y[0]) and x[1].overlaps(y[1]):
                    pairs.append(df)

        ls_all = self._add_solitary_annotations(text, pairs, id_col)
        df_all = pd.concat([x for x in ls_all if not x.empty])
        df_all.reset_index(inplace=True, drop=True)
        df_all.sort_values(["hash", "annotator"], inplace=True)
        return df_all

    def _agreement(
        self,
        df: pd.DataFrame,
        measurement: str,
        by_text: bool = False,
        index: str = None,
    ):
        # set parameters
        msg = f"neither 'rel_id' nor 'id' are in df.columns: {df.columns}"
        annotators = df["annotator"].unique().tolist()
        if "id" in df.columns:
            annotation = "entity"
            column = "label"
            store = "entity_agreement"
        elif "rel_id" in df.columns:
            annotation = "relation"
            column = "type"
            store = "relation_agreement"
        else:
            raise ValueError(msg)

        if measurement == "exact":
            if "id" in df.columns:
                X = (
                    df[df["annotator"] == annotators[0]]["span"]
                    + " = "
                    + df[df["annotator"] == annotators[0]]["label"]
                )
                Y = (
                    df[df["annotator"] == annotators[1]]["span"]
                    + " = "
                    + df[df["annotator"] == annotators[1]]["label"]
                )
            elif "rel_id" in df.columns:
                X = (
                    df[df["annotator"] == annotators[0]]["from_span"]
                    + " -> "
                    + df[df["annotator"] == annotators[0]]["type"]
                    + " -> "
                    + df[df["annotator"] == annotators[0]]["to_span"]
                )
                Y = (
                    df[df["annotator"] == annotators[1]]["from_span"]
                    + " -> "
                    + df[df["annotator"] == annotators[1]]["type"]
                    + " -> "
                    + df[df["annotator"] == annotators[1]]["to_span"]
                )
        elif measurement == "overlap":
            X = df[df["annotator"] == annotators[0]][column]
            Y = df[df["annotator"] == annotators[1]][column]
        else:
            raise ValueError(
                f"measurement must be 'exact' or 'overlap, not {measurement}"
            )

        X.reset_index(drop=True, inplace=True)
        Y.reset_index(drop=True, inplace=True)

        out = pd.DataFrame(
            [
                {"metric": metric[0].__name__, "score": metric[0](X, Y, **metric[1])}
                for metric in self.metrics
            ]
        )
        out["annotators"] = "|".join(annotators)
        out["annotation"] = annotation
        out["measurement"] = measurement
        out["support"] = len(df)

        if by_text:
            store += "_by_text"
            out["index"] = index

        setattr(self, store, pd.concat([getattr(self, store), out]))

        if self.f1_classification_report:
            report_store = f"{store}_report"
            report = sklearn.metrics.classification_report(
                X,
                Y,
                output_dict=True,
                zero_division=0,
            )
            report = pd.DataFrame(report).transpose()
            report["annotation"] = annotation
            report["measurement"] = measurement
            if by_text:
                report["index"] = index

            setattr(
                self, report_store, pd.concat([getattr(self, report_store), report])
            )

    def agreement(self):
        self.reset()
        dfs = [self.entity_match]
        if not self.relation.empty:
            dfs.append(self.relation_match)
        # whole dataset exact
        for df in dfs:
            self._agreement(df, measurement="exact")
        # whole dataset overlap
        for df in dfs:
            self._agreement(df, measurement="overlap")
        # by text exact entity
        for index, df in self.entity_match.groupby("index"):
            self._agreement(df, measurement="exact", by_text=True, index=index)
        # by text overlap entity
        for index, df in self.entity_match.groupby("index"):
            self._agreement(df, measurement="overlap", by_text=True, index=index)

        if not self.relation.empty:
            # by text exact relation
            for index, df in self.relation_match.groupby("index"):
                self._agreement(df, measurement="exact", by_text=True, index=index)
            # by text overlap relation
            for index, df in self.relation_match.groupby("index"):
                self._agreement(df, measurement="overlap", by_text=True, index=index)
            self.relation_match.reset_index(drop=True, inplace=True)

        # reset indexes
        self.entity_agreement.reset_index(drop=True, inplace=True)
        self.relation_agreement.reset_index(drop=True, inplace=True)
        self.entity_agreement_by_text.reset_index(drop=True, inplace=True)
        self.relation_agreement_by_text.reset_index(drop=True, inplace=True)

    @staticmethod
    def _relation_flip_direction(df: pd.DataFrame, new_relation=None):
        df["from_id_new"] = df["to_id"].copy()
        df["to_id_new"] = df["from_id"].copy()
        df["from_span_new"] = df["to_span"].copy()
        df["to_span_new"] = df["from_span"].copy()
        df.drop(["to_id", "from_id", "to_span", "from_span"], axis=1, inplace=True)
        df.rename(
            {
                "from_id_new": "from_id",
                "to_id_new": "to_id",
                "from_span_new": "from_span",
                "to_span_new": "to_span",
            },
            axis=1,
            inplace=True,
        )

        # NOTE applied to llm generated data, deprecate in future
        if "to_label" in df.columns:
            df["from_label_new"] = df["to_label"].copy()
            df["to_label_new"] = df["from_label"].copy()
            df.drop(["to_label", "from_label"], axis=1, inplace=True)
            df.rename(
                {
                    "from_label_new": "from_label",
                    "to_label_new": "to_label",
                },
                axis=1,
                inplace=True,
            )

        if new_relation:
            df["type"] = new_relation
        return df

    def relation_flip_direction(self, old_relation, new_relation=None):
        """Flips the directionality of a relation and renames if `new_relation`.

        Notes:
            Updates the `relation_match` dataframe in place.
        """
        tmp = self.relation.loc[self.relation["type"] == old_relation].copy()
        tmp = self._relation_flip_direction(tmp, new_relation)
        self.relation = self.relation.loc[self.relation["type"] != old_relation].copy()
        self.relation = pd.concat([self.relation, tmp.reorder_levels([1, 0])])

    def reset(self):
        self.entity_agreement = pd.DataFrame()
        self.entity_agreement_by_text = pd.DataFrame()
        self.relation_agreement = pd.DataFrame()
        self.relation_agreement_by_text = pd.DataFrame()
        self.entity_agreement_report = pd.DataFrame()
        self.entity_agreement_by_text_report = pd.DataFrame()
        self.relation_agreement_report = pd.DataFrame()
        self.relation_agreement_by_text_report = pd.DataFrame()

    def save(self):
        names = Path("--".join(self.annotators))
        self.save_dir = self.files[0].parents[0] / Path("annotator-agreement/")
        dataframes = [
            (k, v) for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame)
        ]
        dataframes = [
            (self.save_dir / names / Path(x[0]).with_suffix(".tsv"), x[1])
            for x in dataframes
        ]
        print(f"... saving dataframes")
        dataframes[0][0].parents[0].mkdir(exist_ok=True, parents=True)
        for file, df in dataframes:
            if not df.empty:
                df.to_csv(file, sep="\t", index=True)

    def __repr__(self):
        return f"PairwiseAgreement obj: {len(self.entity)} entitites; {len(self.entity)} relations; {[x.name for x in self.files]}"

    def __init__(
        self,
        file_x: str,
        file_y: str,
        run_match: bool = True,
        metrics: List[tuple[Callable, dict]] = [
            ("f1_score", {"average": "micro"}),
            ("matthews_corrcoef", {}),
        ],
        f1_classification_report: bool = True,
        save: bool = False,
    ):
        self.f1_classification_report = f1_classification_report
        source = []
        self.entity_match = []
        self.relation_match = []
        self.reset()
        # load data
        self.files = [Path(x) for x in [file_x, file_y]]
        self.metrics = [(getattr(sklearn.metrics, x[0]), x[1]) for x in metrics]
        self.annotators = [file.with_suffix("").name for file in self.files]
        for i, file in enumerate(self.files):
            df = pd.read_json(file, lines=True)
            df["annotator"] = file.with_suffix("").name
            source.append(df)

        self.source = pd.concat(source)

        self.source.rename({"id": "text_id"}, axis=1, inplace=True)
        # get annotation types
        self.entity = pd.DataFrame()
        self.relation = pd.DataFrame()
        self.entities()
        if "relations" in self.source.columns:
            self.relations()

        # match annotations and calculate agreement
        if run_match:
            self.match()
            self.agreement()

        if save:
            self.save()


class PairwiseAgreementMany:
    pair: List[PairwiseAgreement]

    def agreement_dfs(self):
        self.entity_agreement = pd.concat([x.entity_agreement for x in self.pair])
        self.relation_agreement = pd.concat([x.relation_agreement for x in self.pair])
        self.entity_agreement.reset_index(drop=True)
        self.relation_agreement.reset_index(drop=True)

    def final_score(self):
        records = []
        names = "|".join([file.with_suffix("").name for file in self.files])
        agreement_dfs = [
            ("entity", self.entity_agreement),
            ("relation", self.relation_agreement),
        ]
        for annotation, agreement_df in agreement_dfs:
            for group in agreement_df.groupby(["metric", "measurement"]):
                metric, measurement = group[0]
                df = group[1]
                dt = dict(
                    annotation=annotation,
                    metric=metric,
                    measurement=measurement,
                    mean=df["score"].mean(),
                    support_sum=df["support"].sum(),
                    support_mean=df["support"].mean(),
                    n_annotators=len(self.files),
                    annotators=names,
                    date=datetime.now().isoformat(timespec="seconds"),
                )
                records.append(dt)
        self.average_pairwise_score = pd.DataFrame.from_records(records)

    def save(self):
        dataframes = [
            (k, v) for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame)
        ]
        dataframes = [
            (self.save_dir / Path(x[0]).with_suffix(".tsv"), x[1]) for x in dataframes
        ]
        print(f"... saving PAM dataframes")
        dataframes[0][0].parents[0].mkdir(exist_ok=True, parents=True)
        for file, df in dataframes:
            if not df.empty:
                df.to_csv(file, sep="\t", index=True)

    def __init__(
        self,
        directory: str,
        run_match: bool = True,
        save=False,
        f1_classification_report: bool = True,
        metrics: List[tuple[Callable, dict]] = [
            ("f1_score", {"average": "micro"}),
            ("matthews_corrcoef", {}),
        ],
    ):
        self.directory = directory
        self.save_dir = Path(self.directory) / Path("annotator-agreement/")
        self.files = [x for x in Path(directory).glob("*.jsonl")]
        self.pair = []
        self.entity_agreement = pd.DataFrame()
        self.relation_agreement = pd.DataFrame()
        if save:
            print(f"... clearing {self.save_dir}")
            shutil.rmtree(self.save_dir, ignore_errors=True)
        for file_x, file_y in itertools.combinations(self.files, 2):
            print(f"... {file_x.name} - {file_y.name}")
            pair = PairwiseAgreement(
                file_x,
                file_y,
                run_match=run_match,
                metrics=metrics,
                save=save,
                f1_classification_report=f1_classification_report,
            )
            self.pair.append(pair)
        self.agreement_dfs()
        self.final_score()
        if save:
            self.save()


@click.command(
    context_settings={"show_default": True},
)
@click.argument(
    "directory", nargs=-1, type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--run/--no-run",
    default=True,
    help="Detect matches and calculate agreement (only instantiates obj if False)",
)
@click.option(
    "--save/--no-save",
    default=False,
    help="Save analysis results to `./agreement/`",
)
@click.option(
    "--f1-report/--no-f1-report",
    default=True,
    help="Produce f1 classification reports for each annotator pair / text",
)
def agree(directory: str, run: bool, save: bool, f1_report):
    """Calculates f1 and matthews_corrcoef pairwise agreement from JSONL annotations"""
    for d in directory:
        msg = f"... calculating agreement for JSONL files in {d}"
        click.echo(msg)
        pam = PairwiseAgreementMany(
            directory=d, run_match=run, save=save, f1_classification_report=f1_report
        )
        pam.agreement_dfs()
        click.echo(f"\nAverage pairwise score: {d}")
        click.echo(pam.average_pairwise_score)


if __name__ == "__main__":
    agree()
