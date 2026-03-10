from graphql_finetuning_pipeline.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_metrics_golden():
    ranked = ["A", "B", "C", "D"]
    assert recall_at_k(ranked, "B", 1) == 0.0
    assert recall_at_k(ranked, "B", 2) == 1.0
    assert mrr_at_k(ranked, "C", 10) == 1.0 / 3.0
    assert ndcg_at_k(ranked, "A", 10) == 1.0
