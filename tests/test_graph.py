"""Unit tests for graph/graph_builder.py.

These tests exercise pure graph-construction logic (no spaCy dependency),
so ``GraphRetriever`` (which needs a live entity extractor) is covered
separately/informally with a real model in integration testing.
"""

from __future__ import annotations

from graph.graph_builder import KnowledgeGraphBuilder, normalize_entity_text


class TestNormalizeEntityText:
    def test_lowercases_and_collapses_whitespace(self) -> None:
        assert normalize_entity_text("  Trastuzumab   IV  ") == "trastuzumab iv"


class TestKnowledgeGraphBuilder:
    def test_add_entities_creates_nodes(self) -> None:
        builder = KnowledgeGraphBuilder()
        entities = [
            {"normalized_text": "trastuzumab", "text": "Trastuzumab", "label": "DRUG", "chunk_id": "c1"},
            {"normalized_text": "breast cancer", "text": "breast cancer", "label": "DISEASE", "chunk_id": "c1"},
        ]
        builder.add_entities(entities)

        assert builder.graph.number_of_nodes() == 2
        assert builder.graph.nodes["trastuzumab"]["mention_count"] == 1

    def test_add_entities_merges_duplicate_mentions(self) -> None:
        builder = KnowledgeGraphBuilder()
        entities = [
            {"normalized_text": "trastuzumab", "text": "Trastuzumab", "label": "DRUG", "chunk_id": "c1"},
            {"normalized_text": "trastuzumab", "text": "trastuzumab", "label": "DRUG", "chunk_id": "c2"},
        ]
        builder.add_entities(entities)

        assert builder.graph.number_of_nodes() == 1
        assert builder.graph.nodes["trastuzumab"]["mention_count"] == 2
        assert builder.graph.nodes["trastuzumab"]["source_chunks"] == {"c1", "c2"}

    def test_add_relations_creates_edge_between_entities(self) -> None:
        builder = KnowledgeGraphBuilder()
        relations = [
            {
                "subject_text": "Trastuzumab",
                "predicate": "treats",
                "object_text": "breast cancer",
                "confidence": 1.0,
                "chunk_id": "c1",
            }
        ]
        builder.add_relations(relations)

        assert builder.graph.has_edge("trastuzumab", "breast cancer")

    def test_add_relations_skips_self_loops(self) -> None:
        builder = KnowledgeGraphBuilder()
        relations = [
            {
                "subject_text": "Trastuzumab",
                "predicate": "co_occurs_with",
                "object_text": "trastuzumab",
                "confidence": 0.4,
                "chunk_id": "c1",
            }
        ]
        builder.add_relations(relations)
        assert builder.graph.number_of_edges() == 0

    def test_get_neighbors_one_hop(self) -> None:
        builder = KnowledgeGraphBuilder()
        builder.add_relations(
            [
                {
                    "subject_text": "trastuzumab",
                    "predicate": "treats",
                    "object_text": "breast cancer",
                    "confidence": 1.0,
                    "chunk_id": "c1",
                }
            ]
        )
        neighbors = builder.get_neighbors("trastuzumab", hops=1)
        assert "breast cancer" in neighbors

    def test_get_neighbors_unknown_node_returns_empty(self) -> None:
        builder = KnowledgeGraphBuilder()
        assert builder.get_neighbors("nonexistent entity") == set()

    def test_stats_returns_expected_keys(self) -> None:
        builder = KnowledgeGraphBuilder()
        stats = builder.stats()
        assert set(stats.keys()) == {"num_nodes", "num_edges", "avg_degree"}
