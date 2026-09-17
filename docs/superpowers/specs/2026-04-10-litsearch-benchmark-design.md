# LitSearch benchmark replacement design

## Goal
Replace the current benchmark data source with LitSearch-derived spatiotemporal queries while keeping the existing prediction format, evaluation flow, and output structure unchanged.

## Scope
This design only changes the benchmark gold dataset source.

It does **not** change:
- `predictions.jsonl` format
- evaluation metric definitions
- evaluation report structure
- run artifact directory layout

## Inputs
- LitSearch query parquet: `../LitSearch/query/full-00000-of-00001.parquet`
- LitSearch clean corpus parquet shards: `../LitSearch/corpus_clean/*.parquet`
- LitSearch S2ORC corpus parquet shards: `../LitSearch/corpus_s2orc/*.parquet`

## Outputs
Create new benchmark files under `data/runs/` without overwriting existing AutoScholarQuery files:
- `LitSearch_spatiotemporal_timeseries_test.jsonl`
- `LitSearch_spatiotemporal_timeseries_dev.jsonl`
- `LitSearch_spatiotemporal_timeseries_nomismatch.jsonl`

Target size:
- test: 10 queries
- dev: 10 queries
- nomismatch: all qualifying no-mismatch queries, or 10 if we keep it intentionally lightweight

## Record format
Each output row must match the existing benchmark schema:

```json
{
  "question": "...",
  "answer": ["Paper A", "Paper B"],
  "answer_arxiv_id": ["2401.12345", "2402.23456"],
  "source_meta": {
    "published_time": "20240101"
  },
  "qid": "LitSearch_test_001"
}
```

## Field mapping
Map LitSearch fields into the benchmark schema as follows:
- `question` <- `query`
- `answer` <- `corpusids` mapped to `corpus_clean.title`
- `answer_arxiv_id` <- `corpusids` mapped to `corpus_s2orc.externalids.arxiv`
- `source_meta.published_time` <- `year` normalized to `YYYY0101`
- `qid` <- `LitSearch_<split>_<index>`

If a paper has no arXiv ID, omit that entry from `answer_arxiv_id` rather than inventing a placeholder.

## Query filtering
We use pure rule-based filtering with no manual review.

### Base eligibility
A LitSearch query is eligible only if all conditions hold:
- `quality >= 2`
- `corpusids` is non-empty
- all required gold papers can be mapped to at least a title
- query text satisfies the spatiotemporal filtering rules below

### Spatiotemporal filtering rules
A query is selected if either of the following holds.

#### Strong-match rules
Keep the query if it contains any of these phrases after case-folding:
- `spatiotemporal`
- `spatio-temporal`
- `spatial-temporal`
- `time series`
- `trajectory forecasting`
- `traffic forecasting`
- `human motion prediction`
- `pose forecasting`

#### Combination rules
Keep the query if either combination is satisfied:
1. one of:
   - `trajectory`
   - `mobility`
   - `traffic`
   - `pose`
   - `human motion`
   and one of:
   - `prediction`
   - `forecasting`
   - `generation`
2. one of:
   - `spatial`
   - `temporal`
   and one of:
   - `prediction`
   - `forecasting`
   - `time series`

## Split construction
After filtering:
1. normalize query text for deterministic processing
2. remove exact duplicate questions if any
3. shuffle with a fixed random seed
4. assign:
   - first 10 to `test`
   - next 10 to `dev`
5. build `nomismatch` from the subset whose gold mappings are complete and stable

If fewer than 20 total eligible queries are available, reduce the split sizes proportionally but preserve deterministic ordering.

## Nomismatch definition
A query belongs in the nomismatch set only when:
- every gold `corpusid` resolves to a title
- the emitted `answer` list is non-empty
- no duplicate or blank paper titles are emitted
- question text is emitted without malformed line wrapping

This keeps the nomismatch benchmark focused on evaluation quality rather than data hygiene problems.

## Validation requirements
The generator must validate each emitted file before completion.

Checks:
- every row is valid JSONL
- every row has `question`, `answer`, `answer_arxiv_id`, `source_meta`, `qid`
- `answer` is non-empty
- `qid` values are unique within each file
- `published_time` is either `YYYY0101` or omitted only if year is unavailable
- all rows can be consumed by the current evaluation workflow without schema changes

## Integration approach
Preferred integration is additive, not destructive.

We will:
- add LitSearch-derived benchmark files beside the current AutoScholarQuery files
- keep existing benchmark files intact for comparison and rollback
- point evaluation runs to the LitSearch files explicitly when desired

We will not:
- overwrite existing AutoScholarQuery benchmark files
- change prediction record schema
- change evaluation metric semantics

## Risks and mitigations

### Risk 1: noisy rule-based selection
Pure keyword filtering may include edge-case queries that are only weakly relevant.

Mitigation:
- use the stronger two-stage rule set rather than a single broad keyword list
- keep the initial split small (10/10) so quality can be inspected quickly

### Risk 2: incomplete metadata mappings
Some LitSearch corpus IDs may lack arXiv IDs or year values.

Mitigation:
- require title resolution for all kept gold papers
- allow empty `answer_arxiv_id` entries where arXiv IDs are unavailable
- derive `published_time` from year only when available

### Risk 3: benchmark instability
Different random selections would make comparisons noisy.

Mitigation:
- use a fixed random seed
- keep deterministic filtering and ordering
- encode split identity directly in `qid`

## Testing plan
Validation should cover:
1. schema-level checks on all emitted rows
2. deterministic split generation given the same input and seed
3. successful loading by the current evaluation path
4. spot verification that selected queries are genuinely from the intended spatiotemporal slice

## Success criteria
The design is successful if:
- LitSearch-derived benchmark files are generated in the existing schema
- current prediction files can be evaluated against them without changing prediction format
- generated files are deterministic and JSONL-valid
- test/dev sets remain lightweight at roughly 10 queries each
- existing AutoScholarQuery benchmark files remain untouched

## Out of scope
This design does not include:
- changing precision/recall definitions
- modifying prediction generation prompts
- introducing semantic or learned query classification
- adding manual review workflows
- replacing the entire evaluation pipeline
