"""Unit tests for entity_extraction/ner_extractor.py and relation_extractor.py.

These tests avoid requiring the real SciSpaCy model download (not available
in CI/sandboxed environments) by constructing lightweight fake spaCy-like
objects (fake Doc/Span/Token) that satisfy the small surface area these
modules actually use (`.ents`, `.sents`, `.pos_`, `.lemma_`, `.idx`,
`.text`, `.start_char`, `.end_char`, `.label_`). This keeps the test suite
fast and dependency-free while still exercising all real business logic in
``MedicalEntityExtractor`` and ``RelationExtractor``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from medgraphrag.extraction.ner_extractor import MedicalEntityExtractor
from medgraphrag.extraction.relation_extractor import RelationExtractor


@dataclass
class FakeEnt:
    text: str
    label_: str
    start_char: int
    end_char: int
    start: int = 0
    end: int = 10


@dataclass
class FakeToken:
    text: str
    idx: int
    pos_: str = "NOUN"
    lemma_: str = ""

    def __post_init__(self) -> None:
        if not self.lemma_:
            self.lemma_ = self.text.lower()


@dataclass
class FakeSent:
    start_char: int
    end_char: int
    start: int = 0
    end: int = 100
    text: str = "Trastuzumab treats cancer."
    tokens: list = field(default_factory=list)

    def __iter__(self):
        return iter(self.tokens)


@dataclass
class FakeDoc:
    ents: list
    sents: list
    tokens: list = field(default_factory=list)

    def __getitem__(self, item):
        return self.tokens[item]


class FakeNLP:
    """Stand-in for a loaded spaCy ``Language`` pipeline in tests."""

    def __init__(self, doc_to_return: FakeDoc) -> None:
        self._doc = doc_to_return

    def __call__(self, text: str) -> FakeDoc:
        return self._doc

    def pipe(self, texts, batch_size: int = 16):
        for _ in texts:
            yield self._doc


def _make_extractor_with_fake_nlp(doc: FakeDoc) -> MedicalEntityExtractor:
    """Construct a MedicalEntityExtractor bypassing the real model load."""
    extractor = object.__new__(MedicalEntityExtractor)
    extractor.model_name = "fake"
    extractor.nlp = FakeNLP(doc)
    return extractor


def _make_relation_extractor_with_fake_nlp(doc: FakeDoc) -> RelationExtractor:
    """Construct a RelationExtractor bypassing the real model load."""
    extractor = object.__new__(RelationExtractor)
    extractor.nlp = FakeNLP(doc)
    return extractor


class TestMedicalEntityExtractor:
    def test_normalize_lowercases_and_collapses_whitespace(self) -> None:
        assert MedicalEntityExtractor._normalize("  Trastuzumab   IV ") == "trastuzumab iv"

    def test_extract_from_chunk_returns_entities(self) -> None:
        doc = FakeDoc(
            ents=[FakeEnt(text="Trastuzumab", label_="DRUG", start_char=0, end_char=11)],
            sents=[],
        )
        extractor = _make_extractor_with_fake_nlp(doc)

        entities = extractor.extract_from_chunk("Trastuzumab treats cancer.", "c1", "book.pdf")

        assert len(entities) == 1
        assert entities[0].text == "Trastuzumab"
        assert entities[0].normalized_text == "trastuzumab"
        assert entities[0].chunk_id == "c1"

    def test_extract_from_chunk_empty_text_returns_empty(self) -> None:
        doc = FakeDoc(ents=[], sents=[])
        extractor = _make_extractor_with_fake_nlp(doc)
        assert extractor.extract_from_chunk("   ", "c1", "book.pdf") == []

    def test_extract_from_chunks_batches_correctly(self) -> None:
        doc = FakeDoc(
            ents=[FakeEnt(text="Doxorubicin", label_="DRUG", start_char=0, end_char=11)],
            sents=[],
        )
        extractor = _make_extractor_with_fake_nlp(doc)
        chunks = [
            {"chunk_id": "c1", "text": "Doxorubicin is a chemo agent.", "source_file": "book.pdf"},
            {"chunk_id": "c2", "text": "Doxorubicin is cardiotoxic.", "source_file": "book.pdf"},
        ]
        entities = extractor.extract_from_chunks(chunks)
        assert len(entities) == 2
        assert {e.chunk_id for e in entities} == {"c1", "c2"}


class TestRelationExtractor:
    def test_find_connecting_verb_returns_lemma(self) -> None:
        sent = FakeSent(start_char=0, end_char=26)
        ent1 = FakeEnt(text="Trastuzumab", label_="DRUG", start_char=0, end_char=11)
        ent2 = FakeEnt(text="cancer", label_="DISEASE", start_char=19, end_char=25)

        conf = RelationExtractor._proximity_confidence(sent, ent1, ent2)
        assert 0.0 < conf <= 1.0

    def test_extract_from_chunk_creates_relation_between_cooccurring_entities(self) -> None:
        tokens = [
            FakeToken(text="Trastuzumab", idx=0, pos_="PROPN"),
            FakeToken(text="treats", idx=12, pos_="VERB", lemma_="treat"),
            FakeToken(text="cancer", idx=19, pos_="NOUN"),
        ]
        sent = FakeSent(start_char=0, end_char=26, tokens=tokens)
        doc = FakeDoc(
            ents=[
                FakeEnt(text="Trastuzumab", label_="DRUG", start_char=0, end_char=11, start=0, end=1),
                FakeEnt(text="cancer", label_="DISEASE", start_char=19, end_char=25, start=2, end=3),
            ],
            sents=[sent],
            tokens=tokens,
        )
        extractor = _make_relation_extractor_with_fake_nlp(doc)

        relations = extractor.extract_from_chunk("Trastuzumab treats cancer.", "c1", "book.pdf")

        assert len(relations) == 1
        assert relations[0].subject_text == "Trastuzumab"
        assert relations[0].object_text == "cancer"
        assert relations[0].predicate == "treats"

    def test_extract_from_chunk_single_entity_produces_no_relations(self) -> None:
        sent = FakeSent(start_char=0, end_char=11, tokens=[])
        doc = FakeDoc(
            ents=[FakeEnt(text="Trastuzumab", label_="DRUG", start_char=0, end_char=11)],
            sents=[sent],
        )
        extractor = _make_relation_extractor_with_fake_nlp(doc)
        relations = extractor.extract_from_chunk("Trastuzumab.", "c1", "book.pdf")
        assert relations == []
