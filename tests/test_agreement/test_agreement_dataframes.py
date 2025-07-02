from pathlib import Path

import pandas as pd
import pytest

from lmf.agreement import PairwiseAgreement
from lmf.utils import str_to_interval

test_dir = Path("tests/agreement")
a = test_dir / Path("a.jsonl")
b = test_dir / Path("b.jsonl")
ab_entity = test_dir / Path("ab_entity.tsv")
ab_relation = test_dir / Path("ab_relation.tsv")
ab_entity_match = test_dir / Path("ab_entity_match.tsv")
ab_relation_match = test_dir / Path("ab_relation_match.tsv")
ab_entity_agreement = test_dir / Path("ab_entity_agreement.tsv")
ab_relation_agreement = test_dir / Path("ab_relation_agreement.tsv")
ab_entity_agreement_by_text = test_dir / Path("ab_entity_agreement_by_text.tsv")
ab_relation_agreement_by_text = test_dir / Path("ab_relation_agreement_by_text.tsv")
ab_entity_agreement_by_text_report = test_dir / Path(
    "ab_entity_agreement_by_text_report.tsv"
)
ab_relation_agreement_by_text_report = test_dir / Path(
    "ab_relation_agreement_by_text_report.tsv"
)
ab_relation_flip = test_dir / Path("ab_relation_flip.tsv")


@pytest.fixture
def pa() -> PairwiseAgreement:
    return PairwiseAgreement(
        a,
        b,
        metrics=[
            ("f1_score", {"average": "micro"}),
        ],
    )


def save_dataframes(x):
    x.entity.to_csv(ab_entity, sep="\t", index=True, index_label="interval_index")
    x.relation.to_csv(ab_relation, sep="\t", index=True, index_label="interval_index")
    x.entity_match.to_csv(ab_entity_match, sep="\t", index=False)
    x.relation_match.to_csv(ab_relation_match, sep="\t", index=False)
    x.entity_agreement.to_csv(ab_entity_agreement, sep="\t", index=False)
    x.relation_agreement.to_csv(ab_relation_agreement, sep="\t", index=False)
    x.entity_agreement_by_text.to_csv(
        ab_entity_agreement_by_text, sep="\t", index=False
    )
    x.relation_agreement_by_text.to_csv(
        ab_relation_agreement_by_text, sep="\t", index=False
    )
    x.entity_agreement_by_text_report.to_csv(
        ab_entity_agreement_by_text_report, sep="\t", index_label="i"
    )
    x.relation_agreement_by_text_report.to_csv(
        ab_relation_agreement_by_text_report, sep="\t", index_label="i"
    )
    x.relation_flip_direction("relation_a", "new_relation")
    x.relation.to_csv(
        ab_relation_flip, sep="\t", index=True, index_label="interval_index"
    )


# x = PairwiseAgreement(a, b, metrics=[("f1_score", {"average": "micro"})])
# save_dataframes(x)


def test_entity_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_entity, sep="\t", index_col="interval_index")
    df.index = pd.IntervalIndex([str_to_interval(x) for x in df.index])
    assert df.equals(pa.entity)


def test_relation_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_relation, sep="\t")
    index_a = pd.IntervalIndex([str_to_interval(x) for x in df.index])
    index_b = pd.IntervalIndex([str_to_interval(x) for x in df["interval_index"]])
    df = df.reset_index(drop=True).drop("interval_index", axis=1)
    df.index = index_a
    df.set_index(index_b, append=True, inplace=True)
    assert df.equals(pa.relation)


def test_entity_match_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_entity_match, sep="\t")
    assert df.equals(pa.entity_match)


def test_relation_match_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_relation_match, sep="\t")
    assert df.equals(pa.relation_match)


def test_entity_agreement_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_entity_agreement, sep="\t")
    assert df.equals(pa.entity_agreement)


def test_relation_agreement_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_relation_agreement, sep="\t")
    assert df.equals(pa.relation_agreement)


def test_entity_agreement_by_text_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_entity_agreement_by_text, sep="\t")
    assert df.equals(pa.entity_agreement_by_text)


def test_relation_agreement_by_text_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_relation_agreement_by_text, sep="\t")
    assert df.equals(pa.relation_agreement_by_text)


def test_entity_agreement_by_text_report_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_entity_agreement_by_text_report, sep="\t", index_col="i")
    assert df.equals(pa.entity_agreement_by_text_report)


def test_relation_agreement_by_text_report_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_relation_agreement_by_text_report, sep="\t", index_col="i")
    assert df.equals(pa.relation_agreement_by_text_report)


def test_relation_flip_dataframe(pa: PairwiseAgreement):
    df = pd.read_csv(ab_relation_flip, sep="\t")
    index_a = pd.IntervalIndex([str_to_interval(x) for x in df.index])
    index_b = pd.IntervalIndex([str_to_interval(x) for x in df["interval_index"]])
    df = df.reset_index(drop=True).drop("interval_index", axis=1)
    df.index = index_a
    df.set_index(index_b, append=True, inplace=True)
    pa.relation_flip_direction("relation_a", "new_relation")
    assert df.equals(pa.relation)


def test_no_relations():
    c = test_dir / Path("c.jsonl")
    d = test_dir / Path("d.jsonl")
    pa = PairwiseAgreement(
        c,
        d,
        metrics=[
            ("f1_score", {"average": "micro"}),
        ],
    )
    assert pa.relation.empty
    assert pa.relation_match == []
    assert pa.relation_agreement.empty
    assert pa.relation_agreement_report.empty
    assert pa.relation_agreement_by_text.empty
    assert pa.relation_agreement_by_text_report.empty
