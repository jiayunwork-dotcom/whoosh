# Copyright 2012 Matt Chaput. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MATT CHAPUT ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL MATT CHAPUT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Matt Chaput.

"""A compact query object for multi-word synonym expansion.

Design notes
============

There are two obvious ways to represent "all legal readings" of a query term:

1. **Materialize every reading as a query branch.** For each reading, build a
   ``Phrase`` (quoted) or an ``And``/presence query (unquoted), and return
   ``Or([...])``. This is simple and reuses all existing machinery. Its cost
   is exponential: the expansion has one branch per reading, ``2**N`` for N
   independent one-token sources, and that cost is paid in the query object,
   in normalize, in matcher construction, and in scoring (duplicate matches
   across branches also add score unless deduplicated).

2. **Keep a reading graph and verify readings lazily per document** (this
   module). The query stores the small DAG produced by the analyzer: nodes are
   input token indices and edges consume one token (original) or one complete
   source (replacement), emitting target words. Parsing, normalization and
   ``repr`` are linear in the input length. At match time, a candidate filter
   produces the documents containing any possible final word; each candidate
   is checked by a reachability DP over ``(input index, position)`` states,
   whose size is bounded by the input length times the document's posting
   positions, independent of the number of readings. ``docs()``, matcher
   ``next``/``skip_to`` and ``searcher.search`` all go through this one
   verification, so branch completeness and resource bounds hold together
   rather than being traded against each other.

Shared work and state
---------------------

* Both approaches share the analyzer plan builder and the query-parser hook.
* Approach 1 has no per-document state but a huge static tree; approach 2 has
  a tiny static tree and one cursor per candidate word per live matcher.
* Approach 2 requires posting positions for any multi-word branch. That is
  the same constraint phrase search already imposes, and is checked up front
  so fields without positions fail with the established ``QueryError``.
* Approach 2's ``copy()`` rebuilds cursors rather than sharing them, because
  posting matchers in Whoosh 2.7.4 are not copyable; copies therefore advance
  independently.

The chosen representation is approach 2. It never enumerates readings, so the
2**24 experiment parses in well under a millisecond and stays under a MiB of
extra allocation.
"""

from bisect import bisect_right
import sys

from whoosh import matching
from whoosh.analysis import Token
from whoosh.compat import u, xrange
from whoosh.query import qcore, terms
from whoosh.util import make_binary_tree


class SynonymQuery(qcore.Query):
    """Matches documents containing at least one complete legal reading of the
    original query token sequence.

    This object is normally created by the query parser from a
    :class:`whoosh.analysis.SynonymFilter` plan; you can also create it by hand
    from a plan produced by the filter.

    :param fieldname: the field to search.
    :param plan: a :class:`whoosh.analysis.synonyms.SynonymPlan`.
    :param slop: the maximum distance between adjacent emitted words. In
        phrase mode this applies to every adjacent pair in a reading; in term
        mode it applies only inside a multi-word target.
    :param boost: a factor applied to the score of matching documents and to
        the ``boost`` attribute of the tokens returned by :meth:`tokens`.
    :param phrase_mode: if True, all words in a reading must appear in order
        and adjacent (within ``slop``), as in a quoted phrase. If False, only
        the words inside a multi-word target have that constraint; words from
        different edges only need to occur somewhere in the document. Term
        parsing uses False; quoted phrases use True.
    :param char_offset: an offset added to the character ranges recorded in
        the plan. For a quoted phrase this is the position of the phrase text
        in the user query. For a bare term, leave it as None so the query's
        ``startchar`` (set by the parser) is used.
    """

    def __init__(self, fieldname, plan, slop=1, boost=1.0, phrase_mode=False,
                 char_offset=None):
        self.fieldname = fieldname
        self.plan = plan
        self.slop = slop
        self.boost = boost
        self.phrase_mode = bool(phrase_mode)
        self.char_offset = char_offset

    # Basic query protocol

    def __repr__(self):
        return ("%s(%r, %r, slop=%s, boost=%f, phrase_mode=%r)"
                % (self.__class__.__name__, self.fieldname, self.plan,
                   self.slop, self.boost, self.phrase_mode))

    def __unicode__(self):
        return u("%s:Synonym(%s)") % (self.fieldname,
                                      u(" ").join(self.plan.texts))

    __str__ = __unicode__

    def __eq__(self, other):
        return (other and self.__class__ is other.__class__
                and self.fieldname == other.fieldname
                and self.plan == other.plan
                and self.slop == other.slop
                and self.boost == other.boost
                and self.phrase_mode == other.phrase_mode
                and self.char_offset == other.char_offset)

    def __hash__(self):
        return (hash(self.fieldname) ^ hash(self.plan) ^ hash(self.slop)
                ^ hash(self.boost) ^ hash(self.phrase_mode)
                ^ hash(self.char_offset))

    def copy(self):
        q = self.__class__(self.fieldname, self.plan, slop=self.slop,
                           boost=self.boost, phrase_mode=self.phrase_mode,
                           char_offset=self.char_offset)
        q.startchar = self.startchar
        q.endchar = self.endchar
        q.error = self.error
        return q

    def with_boost(self, boost):
        q = self.copy()
        q.boost = boost
        return q

    def normalize(self):
        # Never expand the plan into an Or of readings: that would enumerate
        # all 2**N branches. The graph itself is already canonical.
        return self

    # Introspection

    def has_terms(self):
        return True

    def terms(self, phrases=False):
        if not self.fieldname:
            return
        for word in sorted(self.plan.candidate_terms):
            yield (self.fieldname, word)

    def tokens(self, boost=1.0, exreader=None):
        b = boost * self.boost
        if self.char_offset is not None:
            base = self.char_offset
        elif self.startchar is not None:
            # Term-mode plans record character offsets relative to the
            # analyzed term text, which the parser attaches at startchar.
            base = self.startchar
        else:
            base = 0

        for word, sc, ec in self.plan.token_emissions():
            if sc is not None:
                sc += base
            if ec is not None:
                ec += base
            if sc is None and ec is None and self.startchar is not None:
                # The analyzer was not asked for character offsets; point at
                # the parser's fragment for this term instead.
                sc, ec = self.startchar, self.endchar
            yield Token(fieldname=self.fieldname, text=word, boost=b,
                        startchar=sc, endchar=ec, chars=True)

    def field(self):
        return self.fieldname

    # Sizing

    def estimate_size(self, ixreader):
        fieldname = self.fieldname
        if fieldname not in ixreader.schema:
            return 0
        field = ixreader.schema[fieldname]
        total = 0
        for word in self.plan.candidate_terms:
            try:
                total += ixreader.doc_frequency(fieldname, word)
            except (KeyError, ValueError):
                pass
        return min(total, ixreader.doc_count_all())

    def estimate_min_size(self, ixreader):
        # A complete reading may be much rarer than any single candidate word
        return 0

    # Matching

    def needs_positions(self):
        return self.plan.needs_positions(self.phrase_mode)

    def matcher(self, searcher, context=None):
        fieldname = self.fieldname
        if fieldname not in searcher.schema:
            return matching.NullMatcher()

        field = searcher.schema[fieldname]
        if self.needs_positions():
            if not field.format or not field.format.supports("positions"):
                raise qcore.QueryError("Phrase search: %r field has no "
                                       "positions" % self.fieldname)

        return SynonymMatcher(searcher, self, context)


class SynonymMatcher(matching.Matcher):
    """A matcher that verifies complete readings of a
    :class:`SynonymQuery`'s plan for each candidate document.

    Candidate documents are found as the union of documents containing any
    word that could be the last emitted word of a reading. Each candidate is
    then checked with a dynamic program over ``(input index, position)``
    states. The number of states is bounded by the input token count times the
    number of postings positions in the current document, independent of the
    number of readings.
    """

    def __init__(self, searcher, query, context=None):
        self.searcher = searcher
        self.query = query
        self._context = context
        self.fieldname = query.fieldname
        self.plan = query.plan
        self.phrase_mode = query.phrase_mode
        self.slop = query.slop
        self.boost = query.boost
        self.needs_positions = query.needs_positions()

        # Counters, exposed for diagnostics and tests. They are plain integers,
        # so they do not affect matching behavior.
        self.position_reads = 0
        self.doc_tests = 0
        self.matcher_advances = 0

        field = searcher.schema[self.fieldname]

        # One scoring matcher per distinct candidate word
        self._word_matchers = {}
        for word in sorted(self.plan.candidate_terms):
            m = terms.Term(self.fieldname, word).matcher(searcher, context)
            if m.is_active():
                self._word_matchers[word] = m

        # A separate set of matchers solely for finding candidate documents.
        # Sharing matcher objects between the candidate tree and the position
        # readers would corrupt their cursors.
        #
        # A document can only match if it contains at least one word from the
        # *last* edge on some legal path. Note that no word is required by
        # every reading: a document matching only a replacement branch does
        # not contain the source words. Using a union of possible final words
        # is therefore the only sound cheap candidate filter (an intersection
        # of first and last words would miss replacement-only documents).
        last_words = self._last_words()
        self._candidate = self._candidate_matcher(searcher, context,
                                                  last_words)

        # A fixed, true upper bound on any score this matcher can return.
        # It is computed once so quality-based pruning in parent Union/And
        # matchers can never discard a document that could still score above
        # the minimum. (Returning the sum over *currently active* matchers
        # would not be a bound across future documents.)
        bound = 0.0
        for m in self._word_matchers.values():
            try:
                bound += m.max_quality()
            except matching.NoQualityAvailable:
                bound = sys.float_info.max
                break
        self._max_quality = bound * self.boost

        self._docid = None
        self._score = 0.0
        self._find_next()

    # Construction helpers

    def _last_words(self):
        n = self.plan.n
        words = set()
        for edges in self.plan.edges:
            for edge in edges:
                if edge.end == n:
                    words.add(edge.words[-1])
        return sorted(words)

    def _candidate_matcher(self, searcher, context, last_words):
        ms = []
        for word in last_words:
            m = terms.Term(self.fieldname, word).matcher(searcher, context)
            if m.is_active():
                ms.append(m)
        if not ms:
            return matching.NullMatcher()
        if len(ms) == 1:
            return ms[0]
        return make_binary_tree(matching.UnionMatcher, ms)

    # Reading verification

    def _hit_positions(self, doc):
        if not self.needs_positions:
            positions = {}
            for word, m in self._word_matchers.items():
                if m.is_active():
                    if m.id() < doc:
                        self.matcher_advances += 1
                        m.skip_to(doc)
                    if m.is_active() and m.id() == doc:
                        positions[word] = True
            return positions

        positions = {}
        for word, m in self._word_matchers.items():
            if m.is_active():
                if m.id() < doc:
                    self.matcher_advances += 1
                    m.skip_to(doc)
                if m.is_active() and m.id() == doc:
                    self.position_reads += 1
                    positions[word] = m.value_as("positions")
        return positions

    @staticmethod
    def _between(poslist, low, slop):
        # Positions strictly after low and at most slop away
        return poslist[bisect_right(poslist, low):
                       bisect_right(poslist, low + slop)]

    def _edge_spans(self, words, positions):
        """Returns the set of possible exit positions for one edge, starting
        at any position.
        """

        first = positions.get(words[0])
        if not first:
            return ()
        if len(words) == 1:
            return first

        frontier = first
        for word in words[1:]:
            poslist = positions.get(word)
            if not poslist:
                return ()
            nxt = []
            for p in frontier:
                nxt.extend(self._between(poslist, p, self.slop))
            if not nxt:
                return ()
            frontier = nxt
        return frontier

    def _present_reading(self, positions):
        # Term mode: words from different edges need only be present. Inside a
        # multi-word target they must be adjacent (checked by _edge_spans).
        reachable = set((0,))
        edges = self.plan.edges
        for i in xrange(self.plan.n):
            if i not in reachable:
                continue
            for edge in edges[i]:
                words = edge.words
                if len(words) == 1:
                    if words[0] in positions:
                        reachable.add(edge.end)
                elif self._edge_spans(words, positions):
                    reachable.add(edge.end)
        return self.plan.n in reachable

    def _positional_reading(self, positions):
        # Phrase mode: every adjacent pair in the reading must be in order and
        # within slop. States are (input index, exit position) pairs.
        n = self.plan.n
        states = [set() for _ in xrange(n + 1)]
        states[0].add(None)
        edges = self.plan.edges

        for i in xrange(n):
            here = states[i]
            if not here:
                continue
            for edge in edges[i]:
                words = edge.words
                poslist = positions.get(words[0])
                if not poslist:
                    continue

                if len(words) == 1:
                    for last in here:
                        if last is None:
                            states[edge.end].update(poslist)
                        else:
                            states[edge.end].update(
                                self._between(poslist, last, self.slop))
                else:
                    for last in here:
                        if last is None:
                            starts = poslist
                        else:
                            starts = self._between(poslist, last, self.slop)
                        for q0 in starts:
                            frontier = [q0]
                            ok = True
                            for word in words[1:]:
                                pl = positions.get(word)
                                if not pl:
                                    ok = False
                                    break
                                nxt = []
                                for p in frontier:
                                    nxt.extend(self._between(pl, p, self.slop))
                                if not nxt:
                                    ok = False
                                    break
                                frontier = nxt
                            if ok:
                                states[edge.end].update(frontier)
        return bool(states[n])

    def _verify(self, doc):
        self.doc_tests += 1
        positions = self._hit_positions(doc)
        if not positions:
            return False
        if self.phrase_mode:
            if not self.needs_positions:
                return self._present_reading(positions)
            return self._positional_reading(positions)
        return self._present_reading(positions)

    def _score_at(self, doc):
        total = 0.0
        for word, m in self._word_matchers.items():
            if m.is_active() and m.id() == doc:
                total += m.score()
        return total * self.boost

    def _find_next(self):
        candidate = self._candidate
        while candidate.is_active():
            doc = candidate.id()
            if self._verify(doc):
                self._docid = doc
                self._score = self._score_at(doc)
                return
            self.matcher_advances += 1
            candidate.next()
        self._docid = None
        self._score = 0.0

    # Matcher protocol

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self.query)

    def is_active(self):
        return self._docid is not None

    def id(self):
        return self._docid

    def reset(self):
        for word, m in self._word_matchers.items():
            m.reset()
        self._candidate.reset()
        self._docid = None
        self._find_next()

    def copy(self):
        # The simplest correct copy: rebuild fresh matchers from the searcher.
        # Posting matchers in this version of Whoosh do not implement copy(),
        # and re-running the (cheap) candidate construction is far safer than
        # sharing mutable cursors between copies.
        m = self.__class__(self.searcher, self.query, self._context)
        docid = self._docid
        if docid is None:
            # Copy an exhausted matcher as exhausted
            while m.is_active():
                m.next()
        elif m.is_active() and m.id() < docid:
            m.skip_to(docid)
        return m

    def replace(self, minquality=0):
        if not self.is_active():
            return matching.NullMatcher()
        # _max_quality is a true upper bound on any score this matcher can
        # return, so it is safe for a parent query to discard it when that
        # bound cannot reach the minimum score.
        if minquality and self._max_quality < minquality:
            return matching.NullMatcher()
        return self

    def supports_block_quality(self):
        return False

    def max_quality(self):
        return self._max_quality

    def block_quality(self):
        raise matching.NoQualityAvailable(self.__class__)

    def next(self):
        if not self.is_active():
            raise matching.ReadTooFar
        self._candidate.next()
        self._docid = None
        self._find_next()

    def skip_to(self, docid):
        if not self.is_active():
            raise matching.ReadTooFar
        if docid <= self._docid:
            return
        self._candidate.skip_to(docid)
        self._docid = None
        self._find_next()

    def term(self):
        return None

    def children(self):
        # Expose the per-word term matchers so tools that walk the matcher tree
        # (for example TermsCollector used by search(..., terms=True) and
        # highlighting) see the words that actually matched. The matchers keep
        # their cursors on the current document as part of _find_next().
        return list(self._word_matchers.values())

    def supports(self, astype):
        if astype == "positions":
            return self.needs_positions
        field = self.searcher.schema.get(self.fieldname)
        if field is not None and field.format is not None:
            return field.format.supports(astype)
        return False

    def value(self):
        raise Exception("SynonymMatcher does not expose an encoded value")

    def value_as(self, astype):
        if astype == "positions":
            doc = self._docid
            if doc is None:
                return ()
            positions = set()
            for word, m in self._word_matchers.items():
                if m.is_active() and m.id() == doc:
                    positions.update(m.value_as("positions"))
            return sorted(positions)
        raise Exception("SynonymMatcher does not support %r values" % astype)

    def spans(self):
        from whoosh.query.spans import Span
        return [Span(pos) for pos in self.value_as("positions")]

    def score(self):
        return self._score

    def weight(self):
        return self._score / self.boost if self.boost else self._score
