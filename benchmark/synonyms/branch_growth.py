"""Branch-growth experiment for query-side multi-word synonym expansion.

Usage, from the repository root::

    PYTHONPATH=src python3 benchmark/synonyms/branch_growth.py

Tiers: 12 and 24 mutually disjoint one-token sources, each with exactly one
two-token target, so the number of complete readings is 2**12 and 2**24.

Expected hit sets come from a *separate, polynomial* reference in this file
(reachability DP). It does not import Whoosh and never enumerates readings; it
is validated against brute-force enumeration on a small tier.

Indexing and document generation happen before ``tracemalloc`` starts. Then the
query is parsed, normalized, and fully consumed via ``searcher.search``,
``Query.docs`` and a matcher loop that reads positions.
"""
from __future__ import print_function

import gc
import random
import sys
import time
import tracemalloc

from whoosh import analysis, fields, qparser
from whoosh.filedb.filestore import RamStorage


# ----------------------------------------------------------------- reference

def _positions(doc):
    pos = {}
    for i, w in enumerate(doc):
        pos.setdefault(w, []).append(i)
    return pos


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


def _chunks_at(qwords, byfirst, i):
    n = len(qwords)
    yield i + 1, (qwords[i],), False
    for src, tgt in byfirst.get(qwords[i], ()):
        k = len(src)
        if i + k <= n and tuple(qwords[i:i + k]) == src:
            yield i + k, tgt, True


def reading_matches(doc, qwords, rules, phrase_mode, slop=1):
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
                if len(words) == 1:
                    ok = words[0] in dset
                else:
                    ok = any(_walk(pos, start, words[1:], slop)
                             for start in pos.get(words[0], ()))
                if ok:
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
                        states[nxt].update(_walk(pos, s, words[1:], slop))
    return bool(states[n])


def enumerate_readings(qwords, rules):
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


def enumerated_matches(doc, qwords, rules, phrase_mode):
    pos = _positions(doc)
    dset = set(doc)
    for reading in enumerate_readings(qwords, rules):
        if phrase_mode:
            words = tuple(w for ch, _ in reading for w in ch)
            if all(w in dset for w in words) and \
               _adjacent_positions(pos, words):
                return True
        else:
            ok = True
            for ch, is_tgt in reading:
                if len(ch) > 1:
                    if not _adjacent_positions(pos, ch):
                        ok = False
                        break
                elif ch[0] not in dset:
                    ok = False
                    break
            if ok:
                return True
    return False


def check_reference_against_enumeration():
    rules = [(("a",), ("x", "y")), (("a", "b"), ("z",)),
             (("b", "c"), ("u", "v")), (("c",), ("a",))]
    docs = [["a", "b", "c"], ["x", "y", "b", "c"], ["z", "c"],
            ["a", "u", "v"], ["u", "v"], ["x", "y"], ["z"],
            ["a", "b", "u", "v"], ["x", "y", "u", "v"]]
    queries = [["a", "b", "c"], ["a", "a"], ["c", "b"], ["a", "c"]]
    for qwords in queries:
        for doc in docs:
            for mode in (True, False):
                fast = reading_matches(doc, qwords, rules, mode)
                slow = enumerated_matches(doc, qwords, rules, mode)
                assert fast == slow, (qwords, doc, mode, fast, slow)
    return True


def _adjacent_positions(pos, words):
    states = set(pos.get(words[0], ()))
    for w in words[1:]:
        nxt = set()
        for b in pos.get(w, ()):
            for a in states:
                if 1 <= b - a <= 1:
                    nxt.add(b)
        states = nxt
        if not states:
            return False
    return True


# ------------------------------------------------------------------ fixture

def make_tier(nsources, ndocs=64, seed=7):
    sources = ["src%02d" % i for i in range(nsources)]
    rules = [((sources[i],), ("tgt%02da" % i, "tgt%02db" % i))
             for i in range(nsources)]
    rnd = random.Random(seed)
    docs = []
    for d in range(ndocs):
        kind = d % 6
        words = []
        if kind == 0:
            words = list(sources)
        elif kind == 1:
            for i in range(nsources):
                words += ["tgt%02da" % i, "tgt%02db" % i]
        elif kind == 2:
            for i in range(nsources):
                if rnd.random() < 0.5:
                    words.append(sources[i])
                else:
                    words += ["tgt%02da" % i, "tgt%02db" % i]
        elif kind == 3:
            mid = nsources // 2
            for i in range(nsources):
                words.append(sources[i] if i != mid else "filler%d" % i)
        elif kind == 4:
            for i in range(nsources):
                if i % 2 == 0:
                    words.append(sources[i])
                else:
                    words += ["tgt%02da" % i, "sep%d" % i, "tgt%02db" % i]
        else:
            half = nsources // 2
            words = list(reversed(sources))
            words += ["tgt%02da" % i for i in range(half, nsources)]
            words += ["tgt%02db" % i for i in range(half, nsources)]
            words.append("trailer")
        while len(words) < 120:
            words.append("pad%03d" % len(words))
        docs.append(words[:120])
    return sources, rules, docs


def build_index(sources, rules, docs):
    ana = (analysis.RegexTokenizer() | analysis.LowercaseFilter()
           | analysis.SynonymFilter(rules))
    schema = fields.Schema(text=fields.TEXT(analyzer=ana, stored=True))
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        for words in docs:
            w.add_document(text=" ".join(words))
    return schema, ix


# ---------------------------------------------------------------- measurement

def measure(nsources, reps=3, ndocs=64):
    sources, rules, docs = make_tier(nsources, ndocs=ndocs)
    schema, ix = build_index(sources, rules, docs)
    qp = qparser.QueryParser("text", schema)
    qstring = '"%s"' % " ".join(sources)

    expected = set(d for d, doc in enumerate(docs)
                   if reading_matches(doc, sources, rules, True))

    rows = []
    for _ in range(reps):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        q = qp.parse(qstring)
        q = q.normalize()
        t1 = time.perf_counter()

        with ix.searcher() as se:
            results = se.search(q, limit=None)
            hits = set(h.docnum for h in results)
            scores = [h.score for h in results]
            t2 = time.perf_counter()

            docs_hits = set(q.docs(se))
            m = q.matcher(se)
            ids = []
            while m.is_active():
                m.value_as("positions")
                ids.append(m.id())
                m.next()
            workload = (m.position_reads, m.doc_tests, m.matcher_advances)
            t3 = time.perf_counter()

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert hits == expected
        assert docs_hits == expected and set(ids) == expected
        assert all(s == s and abs(s) < 1e12 for s in scores)
        rows.append(dict(parse=t1 - t0, search=t2 - t1, docs=t3 - t2,
                         peak=peak, hits=len(hits), qsize=len(repr(q)),
                         workload=workload))
    return rows


if __name__ == "__main__":
    print("tier  rep parse_s search_s docs_s peak_MiB hits q_repr_chars "
          "pos_reads/doc_tests/matcher_advances readings")
    ok = check_reference_against_enumeration()
    print("reference DP == enumeration (small cross-check):", ok)
    for nsources in (12, 24):
        rows = measure(nsources)
        for i, r in enumerate(rows):
            print("%2d %4d %.4f %.4f %.4f %.2f %d %d %d/%d/%d %d"
                  % (nsources, i, r["parse"], r["search"], r["docs"],
                     r["peak"] / 2 ** 20, r["hits"], r["qsize"],
                     r["workload"][0], r["workload"][1], r["workload"][2],
                     2 ** nsources))
        assert max(r["search"] for r in rows) < 10.0
        assert max(r["peak"] for r in rows) < 64 * 2 ** 20
    print("all timing and allocation bounds satisfied")
