# coding=utf-8
"""Cross-check synonym matching against an independent enumerator, and run a
branch-growth/resource experiment.

The enumerator below never imports Whoosh: it brute-forces all complete
readings from the rule list and the query tokens. It is used only for small
cases, where enumeration is affordable.

The scale test at the bottom uses a *polynomial* reference (reachability over
input index / positions), because enumerating 2**24 readings is impossible by
design. The polynomial reference is itself validated against the enumerator on
small inputs (``test_reference_agrees_with_enumeration``).

The 24-source tier takes a few seconds; it is marked ``slow`` so the default
run can skip it with ``-m "not slow"``. Once indexing and documents are built,
timing/allocations start, and the whole query is parsed and fully consumed.
"""

import gc
import random
import tracemalloc

import pytest

from whoosh import analysis, fields, qparser
from whoosh.filedb.filestore import RamStorage


# ---------------------------------------------------------------- reference

def _positions(doc):
    pos = {}
    for i, w in enumerate(doc):
        pos.setdefault(w, []).append(i)
    return pos


def _adjacent(pos, words, slop=1):
    states = set(pos.get(words[0], ()))
    for w in words[1:]:
        nxt = set()
        for b in pos.get(w, ()):
            for a in states:
                if 1 <= b - a <= slop:
                    nxt.add(b)
        states = nxt
        if not states:
            return False
    return True


def _chunks_at(query_tokens, byfirst, i):
    n = len(query_tokens)
    yield i + 1, (query_tokens[i],), False
    for src, tgt in byfirst.get(query_tokens[i], ()):
        k = len(src)
        if i + k <= n and tuple(query_tokens[i:i + k]) == src:
            yield i + k, tgt, True


def reference_reading(doc, qwords, rules, phrase_mode, slop=1):
    pos = _positions(doc)
    dset = set(doc)
    n = len(qwords)
    byfirst = {}
    for src, tgt in rules:
        byfirst.setdefault(src[0], []).append((tuple(src), tuple(tgt)))

    if not phrase_mode:
        reachable = [False] * (n + 1)
        reachable[0] = True
        for i in range(n):
            if not reachable[i]:
                continue
            for nxt, words, _ in _chunks_at(qwords, byfirst, i):
                if (len(words) > 1 and _adjacent(pos, words, slop)) or \
                   (len(words) == 1 and words[0] in dset):
                    reachable[nxt] = True
        return reachable[n]

    states = [set() for _ in range(n + 1)]
    states[0].add(None)
    for i in range(n):
        if not states[i]:
            continue
        for nxt, words, _ in _chunks_at(qwords, byfirst, i):
            pl = pos.get(words[0])
            if not pl:
                continue
            if len(words) == 1:
                for last in states[i]:
                    if last is None:
                        states[nxt].update(pl)
                    else:
                        states[nxt].update(p for p in pl
                                           if 1 <= p - last <= slop)
            else:
                for last in states[i]:
                    starts = pl if last is None else \
                        [p for p in pl if 1 <= p - last <= slop]
                    for s in starts:
                        states[nxt].update(
                            _walk(pos, s, words[1:], slop))
    return bool(states[n])


def _walk(pos, start, words, slop):
    frontier = {start}
    for w in words:
        nxt = set()
        for a in frontier:
            for b in pos.get(w, ()):
                if 1 <= b - a <= slop:
                    nxt.add(b)
        if not nxt:
            return set()
        frontier = nxt
    return frontier


def enumerate_readings(qwords, rules):
    """Brute-force enumeration of complete readings, for small inputs only."""
    n = len(qwords)
    byfirst = {}
    for src, tgt in rules:
        byfirst.setdefault(src[0], []).append((tuple(src), tuple(tgt)))
    out = []

    def rec(i, chunks):
        if i == n:
            out.append(tuple(chunks))
            return
        for nxt, words, is_tgt in _chunks_at(qwords, byfirst, i):
            rec(nxt, chunks + [(words, is_tgt)])

    rec(0, [])
    return out


def enumerated_reading(doc, qwords, rules, phrase_mode, slop=1):
    pos = _positions(doc)
    dset = set(doc)
    for reading in enumerate_readings(qwords, rules):
        if phrase_mode:
            words = tuple(w for ch, _ in reading for w in ch)
            if all(w in dset for w in words) and _adjacent(pos, words, slop):
                return True
        else:
            ok = True
            for ch, is_tgt in reading:
                if len(ch) > 1:
                    if not _adjacent(pos, ch, slop):
                        ok = False
                        break
                elif ch[0] not in dset:
                    ok = False
                    break
            if ok:
                return True
    return False


# ---------------------------------------------------------------- fixtures

def _analyzer(rules):
    return (analysis.RegexTokenizer() | analysis.LowercaseFilter()
            | analysis.SynonymFilter(rules))


def _index(rules, docs):
    schema = fields.Schema(text=fields.TEXT(analyzer=_analyzer(rules),
                                            stored=True))
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        for d in docs:
            w.add_document(text=d)
    return schema, ix


def _hits(schema, ix, q):
    with ix.searcher() as se:
        return set(h.docnum for h in se.search(q, limit=None))


# ---------------------------------------------------------------- cross-check

def test_reference_agrees_with_enumeration():
    rules = [(("a",), ("x", "y")), (("a", "b"), ("z",)),
             (("b", "c"), ("u", "v")), (("c",), ("a",))]
    docs = [
        ["a", "b", "c"], ["x", "y", "b", "c"], ["z", "c"], ["a", "u", "v"],
        ["u", "v"], ["x", "y"], ["z"], ["a", "b", "u", "v"],
    ]
    for qwords in (["a", "b", "c"], ["a", "a"], ["c", "b"], ["a", "c"]):
        for phrase_mode in (True, False):
            for doc in docs:
                a = reference_reading(doc, qwords, rules, phrase_mode)
                b = enumerated_reading(doc, qwords, rules, phrase_mode)
                assert a == b, (qwords, doc, phrase_mode, a, b)


def test_random_cross_check():
    rnd = random.Random(2024)
    vocab = ["a", "b", "c", "d"]
    for trial in range(25):
        # random small rules
        rules = []
        for _ in range(rnd.randint(1, 4)):
            slen = rnd.randint(1, 2)
            tlen = rnd.randint(1, 2)
            src = tuple(rnd.choice(vocab) for _ in range(slen))
            tgt = tuple(rnd.choice(vocab) for _ in range(tlen))
            rules.append((src, tgt))
        qlen = rnd.randint(1, 4)
        qwords = [rnd.choice(vocab) for _ in range(qlen)]
        docs = [[rnd.choice(vocab) for _ in range(rnd.randint(1, 6))]
                for _ in range(8)]

        for phrase_mode in (True, False):
            expected = set(d for d, doc in enumerate(docs)
                           if reference_reading(doc, qwords, rules,
                                                phrase_mode))
            enumerated = set(d for d, doc in enumerate(docs)
                             if enumerated_reading(doc, qwords, rules,
                                                   phrase_mode))
            assert expected == enumerated

            schema, ix = _index(rules, [" ".join(d) for d in docs])
            qp = qparser.QueryParser("text", schema)
            qstring = " ".join(qwords)
            if phrase_mode:
                qstring = '"%s"' % qstring
            else:
                qstring = "-".join(qwords)
            q = qp.parse(qstring)
            got = _hits(schema, ix, q)
            assert got == expected, (trial, rules, qwords, phrase_mode,
                                     got, expected)


# ---------------------------------------------------------------- scale

def _tier(nsources, ndocs=64, seed=7):
    sources = ["src%02d" % i for i in range(nsources)]
    rules = [((sources[i],), ("tgt%02da" % i, "tgt%02db" % i))
             for i in range(nsources)]
    rnd = random.Random(seed)
    docs = []
    for d in range(ndocs):
        kind = d % 6
        words = []
        if kind == 0:                      # all original: matches
            words = list(sources)
        elif kind == 1:                    # all replaced: matches
            for i in range(nsources):
                words += ["tgt%02da" % i, "tgt%02db" % i]
        elif kind == 2:                    # mixed legal readings: matches
            for i in range(nsources):
                if rnd.random() < 0.5:
                    words.append(sources[i])
                else:
                    words += ["tgt%02da" % i, "tgt%02db" % i]
        elif kind == 3:                    # deletion: no complete reading
            mid = nsources // 2
            for i in range(nsources):
                words.append(sources[i] if i != mid else "filler%d" % i)
        elif kind == 4:                    # split targets: no reading
            for i in range(nsources):
                if i % 2 == 0:
                    words.append(sources[i])
                else:
                    words += ["tgt%02da" % i, "sep%d" % i, "tgt%02db" % i]
        else:                              # incompatible concatenation
            half = nsources // 2
            words = list(reversed(sources))
            words += ["tgt%02da" % i for i in range(half, nsources)]
            words += ["tgt%02db" % i for i in range(half, nsources)]
            words.append("trailer")
        while len(words) < 120:
            words.append("pad%03d" % len(words))
        docs.append(words[:120])
    return sources, rules, docs


def _run_tier(nsources, reps=3):
    sources, rules, docs = _tier(nsources)
    schema, ix = _index(rules, docs)
    qp = qparser.QueryParser("text", schema)
    qstring = '"%s"' % " ".join(sources)

    expected = set()
    qwords = list(sources)
    for dnum, doc in enumerate(docs):
        if reference_reading(doc, qwords, rules, True):
            expected.add(dnum)

    rows = []
    for _ in range(reps):
        gc.collect()
        tracemalloc.start()
        try:
            start = __import__("time").perf_counter()
            q = qp.parse(qstring)
            q = q.normalize()
            parsed = __import__("time").perf_counter()

            with ix.searcher() as se:
                results = se.search(q, limit=None)
                hits = set(h.docnum for h in results)
                scores = [h.score for h in results]
                matched = __import__("time").perf_counter()

                docs_hits = set(q.docs(se))
                m = q.matcher(se)
                ids = []
                while m.is_active():
                    m.value_as("positions")
                    ids.append(m.id())
                    m.next()
                workload = (m.position_reads, m.doc_tests,
                            m.matcher_advances)
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        assert hits == expected
        assert docs_hits == expected
        assert set(ids) == expected
        assert all(s == s and abs(s) < 1e12 for s in scores)
        rows.append(dict(parse=parsed - start, match=matched - parsed,
                         match2=__import__("time").perf_counter() - matched,
                         peak=peak, hits=len(hits),
                         qsize=len(repr(q)), workload=workload))
    return rows


@pytest.mark.slow
def test_branch_growth_12():
    rows = _run_tier(12)
    for r in rows:
        assert r["match"] < 10.0
        assert r["peak"] < 64 * 2 ** 20


@pytest.mark.slow
def test_branch_growth_24():
    rows = _run_tier(24)
    for r in rows:
        assert r["match"] < 10.0
        assert r["peak"] < 64 * 2 ** 20
        # query representation must stay linear in the number of sources,
        # far smaller than any enumeration of 2**24 readings
        assert r["qsize"] < 5000
