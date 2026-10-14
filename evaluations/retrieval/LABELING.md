# Retrieval dataset labeling guide

## Unit of annotation

Each row is one user question. `gold_chunk_ids` contains every corpus chunk that directly
supports a correct answer; `gold_source_ids` contains the corresponding public documentation
source IDs. IDs are stable and must not be replaced with URLs in the query file.

`answerable` is true only when the checked-in corpus contains enough evidence to answer the
question without outside knowledge. Unanswerable rows have empty gold arrays. `difficulty` is
`easy` for a direct lexical match, `medium` for a paraphrase or two-hop wording, and `hard`
for an indirect or easily confused formulation.

## Inter-annotator procedure

Two annotators independently label the query and then compare chunk and source selections.
Disagreements are resolved by opening the cited Python documentation URL and applying the
support rule above. The adjudicated label is checked into `queries.jsonl`; the corpus and
query checksum in `MANIFEST.json` are updated together. For this v1.0.0 set, all 80
answerable rows have one supporting chunk and all 20 unanswerable rows have no supporting
chunk. A second-pass audit sampled one row per chunk and verified its URL, chunk text, and
gold IDs against the Python 3 documentation pages.

Rows are not labeled for generation quality. Retrieval is correct when at least one gold
chunk appears in the requested cutoff; ranking metrics use the complete gold set.
