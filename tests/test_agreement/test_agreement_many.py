from pathlib import Path

import pandas as pd
import pytest

from lmf.agreement import PairwiseAgreementMany

# NOTE see comments in e.jsonl and f.jsonl for descriptions of each text annotation

test_dir = Path("tests/agreement_many")

# x = PairwiseAgreementMany(test_dir)
# x.agreement_dfs()
# x.entity_agreement.to_csv(test_dir / "entity_agreement.tsv", sep="\t", index=False)
# x.relation_agreement.to_csv(test_dir / "relation_agreement.tsv", sep="\t", index=False)


@pytest.fixture
def pam() -> PairwiseAgreementMany:
    return PairwiseAgreementMany(test_dir)


def test_whole_directory(pam: PairwiseAgreementMany):
    for i, pair in enumerate(pam.pair):
        assert (i, pair.entity_agreement["score"].max()) == (i, 1.0)
