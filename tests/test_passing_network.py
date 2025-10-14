from __future__ import annotations
import pandas as pd
from src.passing_network import NetworkResult


def test_network_result_dataclass():
    n = NetworkResult(nodes=pd.DataFrame(), edges=pd.DataFrame())
    assert hasattr(n, "nodes") and hasattr(n, "edges")
