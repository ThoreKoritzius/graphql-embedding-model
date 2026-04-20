# Real-world GraphQL schemas

Four public schemas kept here so the training pipeline has something other
than synthetic templates to learn from. Each is loaded as a separate
"world" by `graphft generate-openai-seed --real-schemas-dir schemas/real`
and defaults to the `test` split, which means they become part of
`real_schema_eval` — the honest transfer benchmark.

| File | Source | Domain | Types | Fields | License |
|---|---|---|---|---|---|
| `github-ghes.graphql` | GitHub Enterprise Server 3.14 public schema, [docs.github.com/public/ghes-3.14](https://docs.github.com/public/ghes-3.14/schema.docs-enterprise.graphql) | dev-tooling / enterprise | ~1,447 | ~6,901 | GitHub Docs public content |
| `saleor.graphql` | [saleor/saleor @ main](https://github.com/saleor/saleor/blob/main/saleor/graphql/schema.graphql) | headless commerce | ~1,424 | ~6,188 | BSD-3-Clause |
| `shopify-storefront.json` | [Shopify/storefront-api-examples](https://github.com/Shopify/storefront-api-examples/blob/master/react-graphql-client/schema.json) | e-commerce storefront | ~104 | ~356 | MIT |
| `anilist.graphql` | [TehNut/AniLinker @ master](https://github.com/TehNut/AniLinker/blob/master/schema.graphql) (mirror of the live AniList API) | media catalog | ~171 | ~861 | MIT |

Why these four:

- **Coverage.** Dev-tooling (GitHub), commerce (Saleor + Shopify), and
  media/catalog (AniList) — three distinct naming conventions and
  relationship topologies. A retriever that only sees one domain
  overfits its vocabulary.
- **Scale contrast.** Shopify Storefront (~100 types) and AniList
  (~170) stress "small schema, tight discrimination". Saleor and
  GHES (~1,400 each) stress "long tail of near-sibling fields" which
  is exactly where traditional embedders fall over.
- **No auth required.** Everything is fetched from public static URLs.
- **Two serialization formats.** Three SDL files plus one
  introspection JSON, so the ingestion path is exercised end-to-end
  (`parse_schema` routes on file suffix).

## Re-downloading

Each file was fetched once with:

```bash
curl -sSL -o schemas/real/github-ghes.graphql \
  https://docs.github.com/public/ghes-3.14/schema.docs-enterprise.graphql
curl -sSL -o schemas/real/saleor.graphql \
  https://raw.githubusercontent.com/saleor/saleor/main/saleor/graphql/schema.graphql
curl -sSL -o schemas/real/anilist.graphql \
  https://raw.githubusercontent.com/TehNut/AniLinker/master/schema.graphql
curl -sSL -o schemas/real/shopify-storefront.json \
  https://raw.githubusercontent.com/Shopify/storefront-api-examples/master/react-graphql-client/schema.json
```

These schemas drift over time. Re-download when you bump your model
version if you want the latest field set; pin them (commit the file)
if you want reproducible training across runs.

## Avoid

- `octokit/graphql-schema/master/schema.graphql` on GitHub. Rich, but
  contains duplicate field definitions around line 15003
  (`EnterpriseOwnerInfo.repositoryDeployKeySetting`) and fails strict
  SDL validation in `graphql-core`. The GHES docs schema above is a
  clean, single-version alternative.
- Yelp Fusion, Shopify Admin. Public introspection requires an API
  token; useful to add later if you have credentials, but they don't
  belong in an unauthenticated starter set.

## Using them

```bash
# One-off: inspect what the ingestion produces
graphft ingest-real-schemas \
  --sdl-dir schemas/real \
  --out-dir artifacts \
  --version 1
cat artifacts/corpus/types_real_v1.jsonl | wc -l   # ~14,300 fields

# Normal usage: mix into seed generation alongside synthetic worlds
export OPENAI_API_KEY=sk-...
graphft generate-openai-seed \
  --out-dir artifacts --version 1 \
  --config examples/pipeline_config.yaml \
  --real-schemas-dir schemas/real \
  --phrasings-per-target 4
```
