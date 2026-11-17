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

"""Query-side multi-word synonym support.

The :class:`SynonymFilter` in this module is deliberately an *analysis*
component only: it never changes the tokens that are indexed. In ``"query"``
mode it annotates tokens with a :class:`SynonymPlan`, a compact description of
all the ways the token sequence can be read. The query parser (and the
:mod:`whoosh.query.synonyms` module) turn that plan into a query object.

Rules are not analyzed. The caller is responsible for applying the same
case-folding, stemming, etc. to rule strings that the analyzer applies to the
input text.
"""

from whoosh.compat import text_type, string_type
from whoosh.analysis.filters import Filter


# Errors

class SynonymError(ValueError):
    """Raised when a synonym rule is malformed.

    This is a subclass of :class:`ValueError` so callers can catch either the
    specific type or the built-in type.
    """


def _check_token(token, which):
    if not isinstance(token, text_type):
        raise SynonymError("%s tokens must be unicode strings, not %r"
                           % (which, token))
    if not token:
        raise SynonymError("%s tokens may not be empty" % which)


def _token_tuple(value, which):
    # A bare string is almost never what the user meant as a "tuple of tokens"
    if isinstance(value, string_type):
        raise SynonymError("%s must be an iterable of tokens, not a string"
                           % which)
    try:
        tokens = tuple(value)
    except TypeError:
        raise SynonymError("%s must be an iterable of tokens, not %r"
                           % (which, value))
    if not tokens:
        raise SynonymError("%s may not be empty" % which)
    for token in tokens:
        _check_token(token, which)
    return tokens


def _normalize_rules(rules):
    """Validates rules and returns a canonical, order-independent table.

    The result is a sorted tuple of ``(source_tuple, targets_tuple)`` pairs,
    where ``targets_tuple`` is a sorted tuple of target tuples. Duplicate rules
    collapse, and the order in which rules were supplied is not significant.
    """

    table = {}
    try:
        iterator = iter(rules)
    except TypeError:
        raise SynonymError("rules must be an iterable of (source, target) "
                           "pairs")

    for rule in iterator:
        if isinstance(rule, string_type):
            raise SynonymError("each rule must be a (source, target) pair")
        try:
            source, target = rule
        except (TypeError, ValueError):
            raise SynonymError("each rule must be a (source, target) pair, "
                               "not %r" % (rule,))

        source = _token_tuple(source, "source")
        target = _token_tuple(target, "target")
        targets = table.setdefault(source, set())
        targets.add(target)

    return tuple((source, tuple(sorted(targets)))
                 for source, targets in sorted(table.items()))


class SynonymRuleSet(object):
    """An immutable, canonical collection of synonym rules.

    You normally don't need to create this object directly; the
    :class:`SynonymFilter` creates one from the ``rules`` argument.
    """

    def __init__(self, rules):
        self.pairs = _normalize_rules(rules)

        # Index sources by their first token so matching a position is cheap
        byfirst = {}
        for source, targets in self.pairs:
            byfirst.setdefault(source[0], []).append((source, targets))
        # Longest first is only a convenience; all matches are still kept
        for first, entries in byfirst.items():
            entries.sort(key=lambda pair: (len(pair[0]), pair[0]))
        self.byfirst = byfirst

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.pairs)

    def __eq__(self, other):
        return (other and self.__class__ is other.__class__
                and self.pairs == other.pairs)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.pairs)

    def matches(self, texts, i):
        """Yields ``(source, targets, endindex)`` for every source matching
        ``texts[i:]``. All matching sources, of every length, are returned.
        """

        for source, targets in self.byfirst.get(texts[i], ()):
            end = i + len(source)
            if end <= len(texts) and tuple(texts[i:end]) == source:
                yield source, targets, end


# Plan objects (pure data; no dependency on whoosh.query)

class SynonymEdge(object):
    """A transition in a :class:`SynonymPlan` graph.

    An edge from input token index ``i`` to ``j`` emits ``words``. If
    ``replacement`` is False, it consumes exactly one input token and emits
    that token unchanged. Otherwise it consumes a complete source sequence and
    emits one of its targets.
    """

    __slots__ = ("start", "end", "words", "replacement",
                 "startchar", "endchar")

    def __init__(self, start, end, words, replacement, startchar=None,
                 endchar=None):
        self.start = start
        self.end = end
        self.words = tuple(words)
        self.replacement = replacement
        self.startchar = startchar
        self.endchar = endchar

    def __repr__(self):
        return ("SynonymEdge(%r, %r, %r, replacement=%r)"
                % (self.start, self.end, self.words, self.replacement))

    def __eq__(self, other):
        return (other and self.__class__ is other.__class__
                and self.start == other.start and self.end == other.end
                and self.words == other.words
                and self.replacement == other.replacement
                and self.startchar == other.startchar
                and self.endchar == other.endchar)

    def __hash__(self):
        return hash((self.start, self.end, self.words, self.replacement,
                     self.startchar, self.endchar))


class SynonymPlan(object):
    """A compact description of all legal readings of a token sequence.

    The plan is a DAG: node 0 is the start, node ``len(tokens)`` is the
    accepting node. An edge ``i -> j`` consumes input tokens ``i`` through
    ``j - 1`` and emits a sequence of output words. Because an edge always
    moves forward, the graph is acyclic, readings always cover the whole input
    exactly once, and generated words are never fed back into the rules.
    """

    def __init__(self, tokeninfos, ruleset):
        """
        :param tokeninfos: a sequence of ``(text, pos, startchar, endchar)``
            tuples, one for each original input token.
        :param ruleset: a :class:`SynonymRuleSet`.
        """

        self.tokens = tuple(tokeninfos)
        self.texts = tuple(info[0] for info in self.tokens)
        self.ruleset = ruleset
        self.n = len(self.tokens)

        edges = []
        for i in range(self.n):
            here = []
            text, pos, sc, ec = self.tokens[i]
            # The original token is always an option, even when a rule also
            # matches here. Keeping it first makes the plan deterministic.
            here.append(SynonymEdge(i, i + 1, (text,), False, sc, ec))

            for source, targets, end in ruleset.matches(self.texts, i):
                ssc, sec = _char_span(self.tokens, i, end)
                for target in targets:
                    here.append(SynonymEdge(i, end, target, True, ssc, sec))

            # Deduplicate edges that emit exactly the same words (for example
            # a duplicate target, or a self-mapping rule). This is important
            # for score invariance: duplicate rules may not add branches.
            deduped = []
            seen = set()
            for edge in here:
                key = (edge.end, edge.words)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(edge)
            # Deterministic order: the original reading first, then
            # replacements ordered by span and words.
            deduped.sort(key=lambda e: (e.replacement, e.end, e.words))
            edges.append(tuple(deduped))

        self.edges = tuple(edges)
        # Only count a replacement if a surviving edge actually replaces
        self.replacement = any(edge.replacement
                               for edges in self.edges for edge in edges)
        self.candidate_terms = self._collect_terms()
        self._signature = (self.texts, self.ruleset.pairs,
                           tuple(tuple((e.start, e.end, e.words,
                                        e.replacement) for e in edges)
                                 for edges in self.edges))

    def _collect_terms(self):
        terms = set()
        for edges in self.edges:
            for edge in edges:
                terms.update(edge.words)
        return frozenset(terms)

    def __repr__(self):
        return ("SynonymPlan(%r, %r)"
                % (self.texts, self.ruleset))

    def __eq__(self, other):
        return (other and self.__class__ is other.__class__
                and self._signature == other._signature)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._signature)

    def edges_from(self, i):
        return self.edges[i]

    def has_multiword_edges(self):
        for edges in self.edges:
            for edge in edges:
                if len(edge.words) > 1:
                    return True
        return False

    def needs_positions(self, phrase_mode=False):
        """Returns True if matching this plan needs posting positions.

        Phrase mode needs positions whenever more than one word can be emitted
        by a reading. Term mode only needs them for multi-word targets.
        """

        if phrase_mode:
            if self.n > 1:
                return True
            # One input token: only a multi-word target needs positions
            return any(len(edge.words) > 1
                       for edge in self.edges[0])
        return self.has_multiword_edges()

    def token_emissions(self):
        """Yields ``(text, startchar, endchar)`` for every word that can be
        emitted, using the original source span for replacement targets.

        Used by query ``tokens()`` so character ranges always point at the
        fragment of the user's query that triggered the replacement.
        """

        yielded = set()
        for i, edges in enumerate(self.edges):
            for edge in edges:
                for word in edge.words:
                    key = (word, edge.startchar, edge.endchar)
                    if key in yielded:
                        continue
                    yielded.add(key)
                    yield key


def _char_span(tokeninfos, start, end):
    first = tokeninfos[start]
    last = tokeninfos[end - 1]
    return first[2], last[3]


# Helpers for the query parser

def has_synonym_filter(analyzer, _seen=None):
    """Returns True if the given analyzer contains a
    :class:`SynonymFilter` anywhere in its composition.

    This lets the parser avoid materializing token lists for analyzers that
    cannot produce synonym plans, preserving the previous fast path and
    behavior for every existing schema. It looks into the common composition
    containers (``CompositeAnalyzer.items`` and ``TeeFilter.filters``).
    Custom analyzer wrappers that hide the filter are not detected; use the
    filter at the top level or inside a CompositeAnalyzer.
    """

    if analyzer is None:
        return False
    if isinstance(analyzer, SynonymFilter):
        return True
    if _seen is None:
        _seen = set()
    if id(analyzer) in _seen:
        return False
    _seen.add(id(analyzer))
    for attr in ("items", "filters"):
        children = getattr(analyzer, attr, None)
        if not children:
            continue
        try:
            iterator = iter(children)
        except TypeError:
            continue
        for child in iterator:
            if has_synonym_filter(child, _seen):
                return True
    return False


def get_synonym_plan(tokens):
    """Returns the common :class:`SynonymPlan` attached to a token list, or
    ``None``. Raises ``ValueError`` if the tokens disagree.
    """

    plan = None
    for token in tokens:
        p = getattr(token, "synonym_plan", None)
        if p is None:
            continue
        if plan is None:
            plan = p
        elif plan is not p and plan != p:
            raise ValueError("token stream contains conflicting synonym plans")
    return plan


# The filter

class SynonymFilter(Filter):
    """Adds query-time multi-word synonym expansion.

    This filter is meant for the query side only. Put it at the end of the
    query analyzer chain, after lowercasing/stemming (the filter does not
    analyze rule strings)::

        ana = RegexTokenizer() | LowercaseFilter() | SynonymFilter([
            (("nyc",), ("new", "york")),
            (("new", "york"), ("nyc",)),
        ])
        schema = Schema(text=TEXT(analyzer=ana))

    In ``"index"`` mode (and any mode other than ``"query"``) the filter is a
    pass-through: input tokens are never rewritten.

    In ``"query"`` mode it annotates tokens with a :class:`SynonymPlan`
    describing *all* legal readings of the input token sequence. The query
    parser turns that plan into a single compact query object. If no rule
    matches, the tokens are returned unchanged, so queries are byte-for-byte
    compatible with the same schema without this filter.

    ``rules`` is an iterable of ``(source_tokens, target_tokens)`` pairs, where
    each side is a non-empty sequence of non-empty unicode strings. A source may
    have several targets. Duplicate rules and rule order are insignificant.
    Malformed rules raise :class:`ValueError` (specifically
    :class:`SynonymError`) when the filter is constructed.

    Replacements are only computed over the original input tokens. Generated
    words are not re-analyzed, so cyclic or self-referential rules terminate.
    All overlapping sources are kept; the filter does not prefer only the
    longest match.
    """

    is_morph = False

    def __init__(self, rules):
        self.ruleset = SynonymRuleSet(rules)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.ruleset.pairs)

    def __eq__(self, other):
        return (other and self.__class__ is other.__class__
                and self.ruleset == other.ruleset)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, tokens):
        return self._filter(tokens)

    def _filter(self, tokens):
        from whoosh.compat import next as _next

        iterator = iter(tokens)
        try:
            first = _next(iterator)
        except StopIteration:
            return

        mode = getattr(first, "mode", "") or ""
        if mode != "query":
            # Indexing (or any other mode) must be completely unchanged
            yield first
            for token in iterator:
                yield token
            return

        # Query streams are short; copy them because tokenizers reuse one
        # Token object for performance. Copies also let us attach private
        # attributes without surprising other consumers.
        alltokens = [first.copy()]
        for token in iterator:
            alltokens.append(token.copy())

        plans = self._build_plans(alltokens)
        if plans is None:
            for token in alltokens:
                yield token
            return

        for token, plan, index in plans:
            token.synonym_plan = plan
            token.synonym_index = index
            yield token

    def _build_plans(self, tokens):
        # Split the stream into runs of contiguous positions. A gap can only
        # come from an upstream filter (typically a stop filter with
        # renumber=False). Version 1 supports contiguous runs only.
        have_positions = bool(tokens and getattr(tokens[0], "positions", False))
        runs = []
        if have_positions:
            current = [tokens[0]]
            for token in tokens[1:]:
                if token.pos == current[-1].pos + 1:
                    current.append(token)
                else:
                    runs.append(current)
                    current = [token]
            runs.append(current)
        else:
            runs = [list(tokens)]

        if len(runs) > 1:
            # There is a position hole. Plan each run so we can tell whether a
            # real replacement branch is involved. If not, every token is
            # passed through unchanged and the hole is irrelevant.
            plans = []
            for run in runs:
                plans.append((run, self._plan_for(run)))
            if any(plan.replacement for _, plan in plans):
                # Matching a replacement as an adjacent phrase would silently
                # tighten the query, so fail loudly instead.
                from whoosh.query.qcore import QueryError
                raise QueryError(
                    "SynonymFilter: cannot apply synonym replacements across "
                    "non-contiguous token positions (found %d position runs); "
                    "the upstream analyzer removed tokens without renumbering "
                    "positions" % len(runs))
            return None

        run = runs[0]
        plan = self._plan_for(run)
        if not plan.replacement:
            return None
        return [(token, plan, i) for i, token in enumerate(run)]

    def _plan_for(self, run):
        infos = []
        for token in run:
            pos = getattr(token, "pos", None)
            sc = getattr(token, "startchar", None) if token.chars else None
            ec = getattr(token, "endchar", None) if token.chars else None
            infos.append((token.text, pos, sc, ec))
        return SynonymPlan(infos, self.ruleset)
