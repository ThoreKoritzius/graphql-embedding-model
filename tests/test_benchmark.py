from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.synthetic import bootstrap_queries, expand_queries, quality_filter, split_queries
from graphql_finetuning_pipeline.eval.benchmark import run_benchmarks
from graphql_finetuning_pipeline.utils.io import write_jsonl


class _DummyEncoder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        del normalize_embeddings, show_progress_bar
        out = []
        for t in texts:
            vec = [0.0] * 64
            for tok in t.lower().split():
                vec[hash(tok) % 64] += 1.0
            out.append(vec)
        return out


def test_run_benchmarks_smoke(tmp_path: Path):
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)

    seed = bootstrap_queries(corpus, rng_seed=9)
    expanded = expand_queries(seed)
    filtered = quality_filter(expanded, type_names={c.type_name for c in corpus}, leakage_threshold=1.0)
    split_rows = split_queries(filtered, seed=4)

    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir(parents=True)
    write_jsonl(bench_dir / "synthetic_holdout.jsonl", [r.model_dump() for r in split_rows if r.split == "test"])

    out = run_benchmarks(bench_dir, corpus, _DummyEncoder(), tmp_path / "out")
    assert "synthetic_holdout" in out
    assert (tmp_path / "out" / "benchmarks_summary.json").exists()
