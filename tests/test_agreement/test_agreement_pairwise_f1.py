from pathlib import Path

import pandas as pd
import pytest

from lmf.agreement import PairwiseAgreement

# NOTE see comments in e.jsonl and f.jsonl for descriptions of each text annotation

test_dir = Path("tests/agreement")
e = test_dir / Path("e.jsonl")
f = test_dir / Path("f.jsonl")
ef_entity_agreement_by_text = test_dir / Path("ef_entity_agreement_by_text.tsv")
ef_relation_agreement_by_text = test_dir / Path("ef_relation_agreement_by_text.tsv")

# x = PairwiseAgreement(e, f, metrics=[("f1_score", {"average": "micro"})])


@pytest.fixture
def pa() -> PairwiseAgreement:
    return PairwiseAgreement(
        e,
        f,
        metrics=[
            ("f1_score", {"average": "micro"}),
        ],
    )


@pytest.fixture
def entityreference() -> pd.DataFrame:
    return pd.read_csv(ef_entity_agreement_by_text, sep="\t")


@pytest.fixture
def relationreference() -> pd.DataFrame:
    return pd.read_csv(ef_relation_agreement_by_text, sep="\t")


def test_entity_agreement_by_text(pa: PairwiseAgreement, entityreference: pd.DataFrame):
    for i in range(len(pa.entity_agreement_by_text)):
        df_json = pa.entity_agreement_by_text.iloc[i].to_json(force_ascii=False)
        ref_json = entityreference.iloc[i].to_json(force_ascii=False)
        assert df_json == ref_json


def test_relation_agreement_by_text(
    pa: PairwiseAgreement, relationreference: pd.DataFrame
):
    for i in range(len(pa.relation_agreement_by_text)):
        df_json = pa.relation_agreement_by_text.iloc[i].to_json(force_ascii=False)
        ref_json = relationreference.iloc[i].to_json(force_ascii=False)
        assert df_json == ref_json
