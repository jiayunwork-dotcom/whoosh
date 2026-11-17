# Query-side multi-word synonym expansion — implementation report

Whoosh 2.7.4 source tree, Python 3.11.2, no new runtime dependencies.

## What was added

| File | Contents |
| --- | --- |
| `src/whoosh/analysis/synonyms.py` | `SynonymFilter`, `SynonymRuleSet`, `SynonymPlan`, `SynonymEdge`, `SynonymError`, `has_synonym_filter`, `get_synonym_plan` |
| `src/whoosh/query/synonyms.py` | `SynonymQuery`, `SynonymMatcher` |
| `src/whoosh/qparser/default.py` | unquoted-term hook in `QueryParser.term_query` |
| `src/whoosh/qparser/plugins.py` | `PhrasePlugin.PhraseNode.query` hook for quoted phrases |
| `tests/test_synonyms.py` | 47 unit/integration tests |
| `tests/test_synonym_branches.py` | cross-check + 2¹²/2²⁴ branch-growth tests |
| `benchmark/synonyms/branch_growth.py` | standalone measurement harness |
| `docs/source/analysis.rst`, `docs/source/parsing.rst`, `docs/source/api/*.rst` | usage and API docs |

## API

```python
from whoosh import analysis, fields, qparser

rules = [(("nyc",), ("new", "york"))]
ana = (analysis.RegexTokenizer()
       | analysis.LowercaseFilter()
       | analysis.SynonymFilter(rules))
schema = fields.Schema(text=fields.TEXT(analyzer=ana))

parser = qparser.QueryParser("text", schema)
parser.parse("nyc subway")     # -> And([SynonymQuery(...), Term('text','subway')])
parser.parse('"nyc subway"')   # -> SynonymQuery(..., phrase_mode=True)
```

* rules: iterable of `(source_tokens, target_tokens)`, each side a non-empty
  sequence of non-empty unicode strings. Multiple targets per source allowed.
  Duplicate rules / rule order are canonicalized away.
* `ValueError` (`SynonymError`) at construction for malformed rules.
* `mode="index"` (and any mode other than `"query"`) is a strict pass-through.
* Fields without the filter keep the old code path byte-for-byte.

## Chosen representation and the alternative

Two viable organizations were considered:

1. **Expand readings into an `Or` of phrases / presence queries.** Simple and
   reuses existing matchers, but the query object, `normalize`, matcher tree
   and scoring all grow with the number of readings (2^N). It also risks
   double-counting scores across branches.
2. **Keep the analysis-time reading DAG and verify readings lazily.** Chosen.
   * Shared static state: the DAG (one edge per original token plus one per
     matching complete source × target). Linear in input length.
   * Match state: one posting cursor per candidate word, plus a candidate
     cursor; verification is a reachability DP over `(input index, position)`.
   * Position constraint: the DP consumes positions, so slop and ordering are
     enforced exactly per chosen reading, with no cross-branch reuse.

Costs of (2): needs posting positions (same requirement as phrase search,
reported as the established `QueryError`), one cursor per candidate word per
live matcher, and non-trivial matcher code. Benefits: branch completeness and
resource bounds hold simultaneously; no reading is ever dropped to save time.

`copy()` rebuilds cursors (Whoosh 2.7.4 posting matchers are not copyable), so
copies advance independently.

## Correctness evidence

* Independent brute-force enumerator (no Whoosh import) cross-checked against
  the library on random small rule sets and documents, both term and phrase
  mode (`tests/test_synonym_branches.py::test_random_cross_check`, 25 random
  trials × 2 modes).
* Judging examples: `nyc→new york` matches `nyc subway` and
  `new york subway`, not `new subway`; `a b→x y`, `b c→u v` gives readings
  `a b c`, `x y c`, `a u v`, never `x u v`.
* Overlap, multi-target, longer/shorter sources, repeated words, duplicate
  rules, rule order, self-mapping, cycles, field scope, NOT/parentheses, slop,
  boost, empty results, no-position fields, `all_terms`, `tokens()` char
  ranges, `normalize`, `docs`/`matcher`/`search` consistency, matcher
  lifecycle and disk reopen are all covered in `tests/test_synonyms.py`.

## Branch-growth experiment

`PYTHONPATH=src python3 benchmark/synonyms/branch_growth.py`

Timing starts after indexing; the query is parsed, normalized, then consumed
through `search`, `Query.docs` and a matcher loop that reads positions.

Measured on this machine (8 vCPU aarch64 container; the stated target is
2 vCPU, 2 GiB — not reproduced here):

```
tier  rep parse_s search_s docs_s peak_MiB hits q_repr_chars pos_reads/doc_tests/matcher_advances readings
reference DP == enumeration (small cross-check): True
12    0 0.0008 0.0169 0.0243 1.29 33 671 1127/64/1147 4096
12    1 0.0003 0.0130 0.0234 0.52 33 671 1127/64/1147 4096
12    2 0.0003 0.0131 0.0229 0.52 33 671 1127/64/1147 4096
24    0 0.0005 0.0251 0.0478 0.83 33 1247 2272/64/2280 16777216
24    1 0.0005 0.0250 0.0456 0.80 33 1247 2272/64/2280 16777216
24    2 0.0005 0.0253 0.0457 0.79 33 1247 2272/64/2280 16777216
all timing and allocation bounds satisfied
```

* 24-source tier: parse ≈0.5 ms, full match ≈25 ms (docs+matcher reload ≈46 ms),
  peak extra allocation <1 MiB. Repeated 3× with identical hit sets.
* Query representation size grows linearly (671 → 1247 chars for 12 → 24
  sources), while readings grow 4096 → 16 777 216.
* Position reads/doc tests/advances: 2272 / 64 / 2280 at the 24-source tier;
  work is O(docs × (candidate words + input length × positions)).

## Re-run commands

```bash
# affected regression files
PYTHONPATH=src python3 -m pytest tests/test_analysis.py tests/test_parsing.py \
    tests/test_queries.py tests/test_matching.py tests/test_synonyms.py \
    tests/test_synonym_branches.py -q

# whole suite
PYTHONPATH=src python3 -m pytest tests -q

# branch growth (slow-tier tests are marked slow)
PYTHONPATH=src python3 -m pytest tests/test_synonym_branches.py -q -m slow
PYTHONPATH=src python3 benchmark/synonyms/branch_growth.py
```

Note: `setup.cfg` contains a legacy `[pytest]` section that pytest ≥ 8
rejects. Use `-c` with a minimal ini file, or `-o` overrides, when running the
suite locally.

## Test results (this run)

* Affected regression files: **197 passed** (analysis, parsing, queries,
  matching, new synonym tests).
* Whole suite: **625 passed, 1 failed** — `tests/test_automata.py::
  test_minimize_dfa` fails identically on the unmodified tree (`git stash`
  verified), so it is pre-existing and unrelated.
* Branch-growth tests (`-m slow`) pass with both timing and allocation
  assertions, three repetitions each.

## Unverified / limitations

* The 2 vCPU / 2 GiB constraint was not reproduced; measurements are from an
  8 vCPU aarch64 container. Margins are ~400× under the 10 s limit and ~75×
  under the 64 MiB limit, so the constraint is very likely met, but it was not
  directly observed.
* Only the contiguous-run case is implemented; position holes with an actual
  replacement raise `QueryError`. Mixed cases (a hole in one parser term, a
  valid replacement in another) raise for the whole query.
* `max_quality()` returns a conservative finite bound. It can be very large
  (`sys.float_info.max`) when a sub-matcher cannot provide quality, which
  disables pruning but never drops results.
* Highlighting works via `matcher.children()` exposing per-word term matchers;
  term-level highlighting is exact, but a multi-word target is not merged into
  one highlighted phrase fragment.
* Custom analyzer wrappers that hide `SynonymFilter` behind an object without
  `items`/`filters` are not detected by `has_synonym_filter`.
