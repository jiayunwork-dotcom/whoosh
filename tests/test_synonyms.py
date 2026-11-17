# coding=utf-8
"""Tests for query-side multi-word synonym expansion."""

from __future__ import with_statement

import pytest

from whoosh import analysis, fields, qparser, query
from whoosh.analysis.synonyms import (SynonymFilter, SynonymError,
                                      get_synonym_plan)
from whoosh.compat import u
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import QueryParser
from whoosh.query.qcore import QueryError
from whoosh.query.synonyms import SynonymQuery, SynonymMatcher


# --------------------------------------------------------------- helpers

def analyzer(rules, stoplist=None, renumber=True):
    chain = (analysis.RegexTokenizer()
             | analysis.LowercaseFilter())
    if stoplist is not None:
        chain = chain | analysis.StopFilter(stoplist=stoplist,
                                            renumber=renumber)
    return chain | SynonymFilter(rules)


def make_index(rules, docs, stoplist=None, renumber=True, phrase=True,
               stored=True, chars=False, fields_=None):
    ana = analyzer(rules, stoplist=stoplist, renumber=renumber)
    if fields_ is None:
        schema = fields.Schema(text=fields.TEXT(analyzer=ana, phrase=phrase,
                                                stored=stored, chars=chars))
    else:
        schema = fields_
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        for doc in docs:
            if isinstance(doc, dict):
                w.add_document(**doc)
            else:
                w.add_document(text=doc)
    return schema, ix


def texts(schema, ix, q):
    with ix.searcher() as se:
        return sorted(se.stored_fields(d)["text"] for d in q.docs(se))


# --------------------------------------------------------------- rule API

def test_rule_validation():
    good = [(("nyc",), ("new", "york")),
            (("new", "york"), ("nyc",)),
            (("a", "b"), ("x", "y")),
            (("a",), ("a",))]
    analysis.SynonymFilter(good)

    bad = [
        [(("",), ("a",))],                 # empty source token
        [((), ("a",))],                    # empty source tuple
        [(("a",), ())],                    # empty target tuple
        [(("a",), ("",))],                 # empty target token
        [(("a", ""), ("b",))],             # empty token in multi-token source
        [("a", "b")],                      # bare strings, not tuples
        [("a",)],                          # not a pair
        [(("a",), ("b",), ("c",))],        # not a pair
        [((u("a"),), (b"b",))],            # non-unicode token
        [((u("a"),), (["b"],))],           # non-string token
        [None],
    ]
    for rules in bad:
        with pytest.raises(ValueError):
            analysis.SynonymFilter(rules)
        with pytest.raises(SynonymError):
            analysis.SynonymFilter(rules)

    # SynonymError is a ValueError subclass
    assert issubclass(SynonymError, ValueError)


def test_rules_canonical_order_independent():
    r1 = analysis.SynonymFilter([(("b",), ("x",)),
                                 (("a",), ("y", "z")),
                                 (("a",), ("w",))])
    r2 = analysis.SynonymFilter([(("a",), ("w",)),
                                 (("b",), ("x",)),
                                 (("a",), ("y", "z"))])
    assert r1 == r2
    assert hash(r1.ruleset) == hash(r2.ruleset)


def test_index_mode_is_passthrough():
    # Note: tokenizers reuse one Token object, so read .text inside the loop.
    ana = analyzer([(("nyc",), ("new", "york"))])
    seen = []
    for t in ana("NYC Subway", mode="index"):
        assert not hasattr(t, "synonym_plan")
        seen.append(t.text)
    assert seen == ["nyc", "subway"]

    # Default mode is not "query", so also unchanged
    assert [t.text for t in ana("NYC Subway")] == ["nyc", "subway"]


def test_no_matching_rule_no_change():
    ana = analyzer([(("nyc",), ("new", "york"))])
    stream = list(ana("hello there", mode="query"))
    assert [t.text for t in stream] == ["hello", "there"]
    assert all(not hasattr(t, "synonym_plan") for t in stream)


def test_analysis_query_tokens_unchanged():
    ana = analyzer([(("nyc",), ("new", "york"))])
    stream = list(ana("NYC subway", mode="query"))
    # Input tokens themselves are *not* rewritten
    assert [t.text for t in stream] == ["nyc", "subway"]
    plan = get_synonym_plan(stream)
    assert plan is not None
    assert plan.replacement


def test_plan_graph():
    ana = analyzer([(("a", "b"), ("x", "y")),
                    (("b", "c"), ("u", "v"))])
    plan = get_synonym_plan(list(ana("a b c", mode="query")))
    # Two replacement edges out of 0, plus the original
    ends = [(e.end, e.words) for e in plan.edges_from(0)]
    assert (1, ("a",)) in ends
    assert (2, ("x", "y")) in ends
    # From 1: original b plus b c -> u v
    ends = [(e.end, e.words) for e in plan.edges_from(1)]
    assert (2, ("b",)) in ends
    assert (3, ("u", "v")) in ends
    # Generated words are never used as rule input: no edge consumes 2 -> 3
    # based on "y" or "u".
    assert all(not e.words or not e.replacement for e in plan.edges_from(2))


def test_plan_does_not_reapply_to_generated_words():
    # Rule x -> a should not feed back from a -> x y
    ana = analyzer([(("a",), ("x", "y")), (("x",), ("a",))])
    plan = get_synonym_plan(list(ana("a b", mode="query")))
    from_0 = [(e.end, e.words) for e in plan.edges_from(0)]
    assert (1, ("x", "y")) in from_0
    # No multi-token consumption ending past the first token starts here
    assert all(e.end <= 1 for e in plan.edges_from(0))
    assert all(not e.replacement for e in plan.edges_from(1))


def test_plan_self_mapping_terminates():
    ana = analyzer([(("a",), ("a",))])
    stream = list(ana("a b", mode="query"))
    # A self-mapping rule adds no distinct branch, so the filter passes the
    # tokens through and attaches no plan.
    plan = get_synonym_plan(stream)
    assert plan is None
    assert [t.text for t in stream] == ["a", "b"]


def test_plan_cycle_does_not_loop():
    ana = analyzer([(("a",), ("b", "c")), (("b", "c"), ("a",))])
    tokens = list(ana("a", mode="query"))
    plan = get_synonym_plan(tokens)
    assert plan.replacement
    # Only source rules over the original input token were applied
    assert [e.words for e in plan.edges_from(0)] == [("a",), ("b", "c")]


def test_plan_duplicate_rules_collapse():
    ana1 = analyzer([(("a",), ("x", "y"))])
    ana2 = analyzer([(("a",), ("x", "y")), (("a",), ("x", "y"))])
    p1 = get_synonym_plan(list(ana1("a b", mode="query")))
    p2 = get_synonym_plan(list(ana2("a b", mode="query")))
    assert p1 == p2


def test_plan_keeps_overlapping_readings():
    ana = analyzer([(("a", "b"), ("x", "y")), (("b", "c"), ("u", "v"))])
    plan = get_synonym_plan(list(ana("a b c", mode="query")))
    # Both the long match at 0 and the one starting at 1 survive
    assert any(e.end == 2 for e in plan.edges_from(0))
    assert any(e.end == 3 for e in plan.edges_from(1))


def test_plan_positions_and_chars():
    ana = analyzer([(("new", "york"), ("nyc",))])
    tokens = list(ana("x New York y", mode="query", chars=True,
                      positions=True))
    plan = get_synonym_plan(tokens)
    edge = [e for e in plan.edges_from(1) if e.replacement][0]
    # The plan records where the source matched in the *original* text
    assert plan.texts == ("x", "new", "york", "y")
    assert edge.startchar == tokens[1].startchar
    assert edge.endchar == tokens[2].endchar


# --------------------------------------------------------------- plans and query

def test_synonym_query_terms_and_tokens():
    ana = analyzer([(("nyc",), ("new", "york"))])
    plan = get_synonym_plan(list(ana("nyc", mode="query", chars=True)))
    q = SynonymQuery("f", plan)
    assert q.all_terms() == set([("f", "nyc"), ("f", "new"),
                                 ("f", "york")])

    # tokens() uses the original query fragment for replacement targets
    toks = list(q.tokens())
    ranges = {}
    for t in toks:
        ranges.setdefault(t.text, (t.startchar, t.endchar))
    assert ranges["nyc"] == (0, 3)
    assert ranges["new"] == (0, 3)
    assert ranges["york"] == (0, 3)


def test_synonym_query_tokens_offset():
    ana = analyzer([(("nyc",), ("new", "york"))])
    plan = get_synonym_plan(list(ana("nyc", mode="query", chars=True)))
    q = SynonymQuery("f", plan, char_offset=10)
    toks = list(q.tokens())
    assert all(t.startchar == 10 and t.endchar == 13 for t in toks)


def test_synonym_query_normalize_is_stable():
    ana = analyzer([(("a",), ("x", "y")), (("b",), ("u", "v"))])
    plan = get_synonym_plan(list(ana("a b", mode="query")))
    q = SynonymQuery("f", plan)
    n = q.normalize()
    assert n == q
    assert len(repr(n)) < 2000  # not expanded to 2**2 phrases


def test_query_not_merged_across_whitespace():
    rules = [(("a", "b"), ("x", "y"))]
    ana = analyzer(rules)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana))
    qp = QueryParser("text", schema)
    q = qp.parse("a b c")
    # Whitespace-separated terms remain separate parser terms
    assert isinstance(q, query.And)
    assert [type(s) for s in q] == [query.Term, query.Term, query.Term]
    # With hyphens the analyzer sees one term, so the rule can apply
    q = qp.parse("a-b-c")
    assert isinstance(q, SynonymQuery)


def test_query_not_merged_across_boolean():
    rules = [(("a", "b"), ("x", "y"))]
    ana = analyzer(rules)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana))
    qp = QueryParser("text", schema)
    q = qp.parse("a OR b")
    assert isinstance(q, query.Or)
    assert all(isinstance(s, query.Term) for s in q)
    q = qp.parse("a AND b")
    assert isinstance(q, query.And)
    assert all(isinstance(s, query.Term) for s in q)


# --------------------------------------------------------------- matching

def test_nyc_subway_hits():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc subway", "new york subway", "new subway", "nyc bus",
            "new york bus"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc subway")
    assert texts(schema, ix, q) == ["new york subway", "nyc subway"]


def test_overlapping_readings():
    rules = [(("a", "b"), ("x", "y")), (("b", "c"), ("u", "v"))]
    docs = ["a b c", "x y c", "a u v", "x u v", "x y", "u v", "a b u v",
            "x y u v"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    # Term mode: "a b c" reads as original, x y c or a u v. "a b u v" also
    # matches because in term mode single-token words only need to be present.
    q = qp.parse("a-b-c")
    assert texts(schema, ix, q) == ["a b c", "a b u v", "a u v", "x y c"]
    # Quoted phrase: complete readings as in-order phrases
    q = qp.parse('"a b c"')
    assert texts(schema, ix, q) == ["a b c", "a u v", "x y c"]
    # x u v is never a complete reading: no path emits x then u
    q = qp.parse('"x u v"')
    assert texts(schema, ix, q) == ["x u v"]


def test_no_cross_chunk_reading():
    # a b -> x y and b c -> u v must not produce the reading x u v
    rules = [(("a", "b"), ("x", "y")), (("b", "c"), ("u", "v"))]
    docs = ["x u v", "a b c"]
    schema, ix = make_index(rules, docs)
    # Build the plan for a b c and verify "x u v" is not a full reading
    ana = analyzer(rules)
    plan = get_synonym_plan(list(ana("a b c", mode="query")))
    from whoosh.query.synonyms import SynonymMatcher

    q = SynonymQuery("text", plan, phrase_mode=True)
    with ix.searcher() as se:
        m = SynonymMatcher(se, q)
        assert list(m.all_ids()) == [1]


def test_multi_target():
    rules = [(("nyc",), ("new", "york")),
             (("nyc",), ("big", "apple"))]
    docs = ["nyc", "new york", "big apple", "new york city", "big city"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    assert texts(schema, ix, q) == ["big apple", "new york", "new york city",
                                    "nyc"]
    for t in q.tokens():
        if t.text in ("new", "big"):
            assert (t.startchar, t.endchar) == (0, 3)


def test_source_longer_than_target():
    rules = [(("new", "york"), ("nyc",))]
    docs = ["new york subway", "nyc subway", "new subway", "york subway"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse('"new york"')
    assert texts(schema, ix, q) == ["new york subway", "nyc subway"]


def test_source_shorter_than_target():
    rules = [(("nyc",), ("new", "york", "city"))]
    docs = ["nyc", "new york city", "new york", "york city"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    assert texts(schema, ix, q) == ["new york city", "nyc"]


def test_repeated_words():
    rules = [(("a", "a"), ("x", "y"))]
    docs = ["a a", "a", "a a a", "x y", "x y a"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse('"a a"')
    # "a a a" contains the adjacent "a a" reading too
    assert texts(schema, ix, q) == ["a a", "a a a", "x y", "x y a"]
    # Term mode is presence-based: a single "a" satisfies the original
    # reading, and "x y a" contains the x y target.
    q = qp.parse("a-a")
    assert texts(schema, ix, q) == ["a", "a a", "a a a", "x y", "x y a"]


def test_repeated_rules_same_results_and_scores():
    docs = ["nyc subway", "new york subway", "new subway"]
    base = [(("nyc",), ("new", "york"))]
    dup = base + base
    rev = list(reversed(dup))
    scored = []
    for rules in (base, dup, rev):
        schema, ix = make_index(rules, docs)
        qp = QueryParser("text", schema)
        q = qp.parse("nyc subway")
        with ix.searcher() as se:
            r = se.search(q, limit=None)
            scored.append([(h["text"], round(h.score, 6)) for h in r])
    assert scored[0] == scored[1] == scored[2]
    assert scored[0]


def test_rule_order_does_not_change_results():
    docs = ["a b c", "x y c", "a u v"]
    rules1 = [(("a", "b"), ("x", "y")), (("b", "c"), ("u", "v"))]
    rules2 = list(reversed(rules1))
    out = []
    for rules in (rules1, rules2):
        schema, ix = make_index(rules, docs)
        qp = QueryParser("text", schema)
        q = qp.parse('"a b c"')
        with ix.searcher() as se:
            r = se.search(q, limit=None)
            out.append([(h["text"], round(h.score, 6)) for h in r])
    assert out[0] == out[1]


def test_self_mapping_terminates_and_matches():
    rules = [(("nyc",), ("nyc",))]
    docs = ["nyc", "new york"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    assert texts(schema, ix, q) == ["nyc"]


def test_cycle_terminates():
    rules = [(("a",), ("b", "c")), (("b", "c"), ("a",))]
    docs = ["a", "b c", "b", "c"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("a")
    assert texts(schema, ix, q) == ["a", "b c"]


def test_field_scope():
    ana = analyzer([(("nyc",), ("new", "york"))])
    schema = fields.Schema(text=fields.TEXT(analyzer=ana, stored=True),
                           plain=fields.TEXT(stored=True))
    docs = [dict(text="nyc subway", plain="nyc subway"),
            dict(text="new york subway", plain="new york subway")]
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        for doc in docs:
            w.add_document(**doc)
    qp = QueryParser("text", schema)
    q = qp.parse("text:nyc")
    assert isinstance(q, SynonymQuery)
    with ix.searcher() as se:
        assert len(list(q.docs(se))) == 2

    # The plain field has no synonym analyzer; it stays a Term
    q = qp.parse("plain:nyc")
    assert isinstance(q, query.Term)
    with ix.searcher() as se:
        assert len(list(q.docs(se))) == 1


def test_not_and_parentheses():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc subway", "new york subway", "new york city", "nyc bus"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("(nyc NOT subway)")
    assert isinstance(q, query.And)
    assert sorted(h["text"] for h in _search(schema, ix, q)) == \
        ["new york city", "nyc bus"]
    q = qp.parse("nyc OR \"new york\"")
    assert sorted(h["text"] for h in _search(schema, ix, q)) == \
        ["new york city", "new york subway", "nyc bus", "nyc subway"]
    q = qp.parse('(nyc subway) AND "new york"')
    assert sorted(h["text"] for h in _search(schema, ix, q)) == \
        ["new york subway"]


def _search(schema, ix, q):
    # Materialize stored fields while the searcher is open
    with ix.searcher() as se:
        return [dict(h) for h in se.search(q, limit=None)]


def test_slop_semantics():
    rules = [(("a", "b"), ("x", "y"))]
    docs = ["a b", "x y", "x q y", "x q q y", "x q q q y"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    # slop=1 requires adjacency
    assert texts(schema, ix, qp.parse('"a b"')) == ["a b", "x y"]
    # slop applies to adjacent emitted words within the chosen reading
    assert texts(schema, ix, qp.parse('"a b"~2')) == ["a b", "x q y", "x y"]
    assert texts(schema, ix, qp.parse('"a b"~4')) == \
        ["a b", "x q q q y", "x q q y", "x q y", "x y"]


def test_boost_applies_to_both_branches():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc", "new york"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    base = qp.parse("nyc")
    boosted = qp.parse("nyc^4")
    with ix.searcher() as se:
        b = dict((h["text"], h.score) for h in se.search(base, limit=None))
        x = dict((h["text"], h.score) for h in se.search(boosted, limit=None))
    assert set(b) == set(x) == set(["nyc", "new york"])
    for text in b:
        assert abs(x[text] - 4 * b[text]) < 1e-6


def test_empty_result():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["something else", "nyc bus"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc subway")
    assert texts(schema, ix, q) == []
    assert list(_search(schema, ix, q)) == []


def test_no_positions_field_rejected():
    rules = [(("nyc",), ("new", "york"))]
    ana = analyzer(rules)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana, phrase=False))
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        w.add_document(text="nyc subway")
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    with ix.searcher() as se:
        with pytest.raises(QueryError):
            list(q.docs(se))

    # A query with no replacement branch is still a plain term and works
    q = qp.parse("unrelated")
    assert isinstance(q, query.Term)


def test_position_holes_raise_query_error():
    # Stop word removed without renumbering leaves a gap in a single term
    # (the term spans positions 0 and 2) and in a quoted phrase.
    rules = [(("nyc",), ("new", "york"))]
    ana = analyzer(rules, stoplist=["subway"], renumber=False)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana))
    qp = QueryParser("text", schema)
    with pytest.raises(QueryError):
        qp.parse("nyc-subway-bus")
    with pytest.raises(QueryError):
        qp.parse('"nyc subway bus"')


def test_position_hole_without_replacement_is_tolerated():
    rules = [(("zzz",), ("new", "york"))]
    ana = analyzer(rules, stoplist=["subway"], renumber=False)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana))
    qp = QueryParser("text", schema)
    q = qp.parse('"nyc subway bus"')
    assert isinstance(q, query.Phrase)
    assert q.words == ["nyc", "bus"]


def test_renumbered_stopword_query_ok():
    # StopFilter with default renumber=True leaves no holes
    rules = [(("nyc",), ("new", "york"))]
    ana = analyzer(rules, stoplist=["the"], renumber=True)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana, stored=True))
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        w.add_document(text="the nyc subway")
    qp = QueryParser("text", schema)
    q = qp.parse("nyc the subway")
    assert texts(schema, ix, q) == ["the nyc subway"]


def test_quoted_single_token_phrase():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc", "new york", "new york city", "new city"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse('"nyc"')
    assert isinstance(q, SynonymQuery)
    assert q.phrase_mode
    # A one-token phrase reading can be either one word or two adjacent words
    assert texts(schema, ix, q) == ["new york", "new york city", "nyc"]


def test_terms_collector_and_highlight():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc subway", "new york subway"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    with ix.searcher() as se:
        r = se.search(q, limit=None, terms=True)
        assert len(r) == 2
        for h in r:
            frag = h.highlights("text", text=h["text"])
            assert "<b" in frag


def test_search_with_filter_and_sortedby():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc bus", "new york bus", "nyc subway"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    with ix.searcher() as se:
        r = se.search(q, limit=None, filter=query.Term("text", "bus"))
        assert sorted(h["text"] for h in r) == ["new york bus", "nyc bus"]
        # sortedby="text" needs the field to be sortable; use docnum ordering
        # to check the full result set instead.
        r = se.search(q, limit=None)
        assert sorted(h["text"] for h in r) == ["new york bus", "nyc bus",
                                                "nyc subway"]


# --------------------------------------------------------------- lifecycle

def test_docs_matcher_search_consistency():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc subway", "new york subway", "new subway", "new york city",
            "nyc"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc subway")
    with ix.searcher() as se:
        d1 = sorted(q.docs(se))
        d2 = sorted(h.docnum for h in se.search(q, limit=None))
        m = q.matcher(se)
        d3 = list(m.all_ids())
        m = q.matcher(se)
        d4 = []
        while m.is_active():
            d4.append(m.id())
            m.next()
    assert d1 == d2 == d3 == d4


def test_copy_independent():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc one", "new york two", "nyc three", "new york four"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    with ix.searcher() as se:
        m1 = q.matcher(se)
        m2 = m1.copy()
        assert m1.id() == m2.id()
        m1.next()
        assert m2.id() != m1.id() or not m1.is_active()
        m2.skip_to(m1.id())
        assert m2.id() == m1.id()
        assert list(m1.all_ids()) == list(m2.all_ids())


def test_skip_to_and_next():
    rules = [(("nyc",), ("new", "york"))]
    docs = ["nyc a", "new york b", "nyc c", "new york d", "nyc e"]
    schema, ix = make_index(rules, docs)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    with ix.searcher() as se:
        m = q.matcher(se)
        assert m.id() == 0
        m.skip_to(2)
        assert m.id() == 2
        m.skip_to(2)  # no-op
        assert m.id() == 2
        m.next()
        assert m.id() == 3
        m.next()
        assert m.id() == 4
        m.next()
        assert not m.is_active()
        with pytest.raises(Exception):
            m.next()


def test_all_terms_finds_candidates():
    rules = [(("nyc",), ("new", "york")),
             (("nyc",), ("big", "apple"))]
    ana = analyzer(rules)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana))
    qp = QueryParser("text", schema)
    q = qp.parse("nyc")
    assert q.all_terms() == set([("text", "nyc"), ("text", "new"),
                                 ("text", "york"), ("text", "big"),
                                 ("text", "apple")])


# --------------------------------------------------------------- disk index

@pytest.mark.parametrize("segments", [1, 3])
def test_multi_segment(tmpdir, segments):
    from whoosh.filedb.filestore import FileStorage
    rules = [(("nyc",), ("new", "york"))]
    ana = analyzer(rules)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana, stored=True))
    st = FileStorage(str(tmpdir))
    ix = st.create_index(schema)
    docs = ["nyc subway", "new york subway", "new york city", "new subway"]
    for start in range(0, len(docs), segments):
        with ix.writer() as w:
            for d in docs[start:start + segments]:
                w.add_document(text=d)
    qp = QueryParser("text", schema)
    q = qp.parse("nyc subway")
    assert texts(schema, ix, q) == ["new york subway", "nyc subway"]
    with ix.searcher() as se:
        m = q.matcher(se)
        assert set(m.all_ids()) == set(q.docs(se))
    st.close()


def test_disk_reopen(tmpdir):
    from whoosh.filedb.filestore import FileStorage
    rules = [(("nyc",), ("new", "york"))]
    ana = analyzer(rules)
    schema = fields.Schema(text=fields.TEXT(analyzer=ana, stored=True))
    path = str(tmpdir)
    st = FileStorage(path)
    ix = st.create_index(schema)
    with ix.writer() as w:
        for d in ["nyc subway", "new york subway", "new subway"]:
            w.add_document(text=d)
    del ix
    ix = st.open_index()
    qp = QueryParser("text", schema)
    q = qp.parse("nyc subway")
    assert texts(schema, ix, q) == ["new york subway", "nyc subway"]
    st.close()
