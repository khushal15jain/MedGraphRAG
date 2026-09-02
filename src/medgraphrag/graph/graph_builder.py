"""Stage 7: Knowledge Graph Construction.

Builds a medical knowledge graph with NetworkX (chosen explicitly over Neo4j
per project constraints — an in-memory graph library is more than sufficient
at the scale of six textbooks and avoids running a separate database
process on an 8GB machine).

Graph schema:
  - Nodes: normalized entity text (e.g. "trastuzumab"), with attributes
    ``label`` (entity type), ``mention_count``, and ``source_chunks`` (the
    set of chunk_ids where this entity was mentioned — this is what makes
    graph retrieval able to map back to retrievable text).
  - Edges: (subject_node, object_node) with attributes ``predicate``,
    ``confidence`` (max confidence seen across mentions), and ``weight``
    (co-occurrence count, used for graph-expansion ranking).

Entity nodes are merged by normalized (lowercased, whitespace-collapsed)
text — a simple but effective form of medical entity linking at this scale;
a production system would layer UMLS/SciSpaCy's ``EntityLinker`` on top,
which this module's ``normalized_text`` field is designed to support later
without a schema change.
"""

from __future__ import annotations

import networkx as nx

from medgraphrag.utils.exceptions import GraphConstructionError
from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    """Constructs and incrementally updates a NetworkX medical knowledge graph."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_entities(self, entities: list[dict]) -> None:
        """Add/merge entity nodes into the graph.

        Args:
            entities: List of entity dicts with keys ``normalized_text``,
                ``text``, ``label``, ``chunk_id``.

        Raises:
            GraphConstructionError: If a required key is missing.
        """
        try:
            for ent in entities:
                node_id = ent["normalized_text"]
                if not node_id:
                    continue

                if self.graph.has_node(node_id):
                    node = self.graph.nodes[node_id]
                    node["mention_count"] += 1
                    node["source_chunks"].add(ent["chunk_id"])
                    node["surface_forms"].add(ent["text"])
                else:
                    self.graph.add_node(
                        node_id,
                        label=ent["label"],
                        mention_count=1,
                        source_chunks={ent["chunk_id"]},
                        surface_forms={ent["text"]},
                    )
        except KeyError as exc:
            raise GraphConstructionError(f"Entity dict missing required key: {exc}") from exc

        logger.info(f"Graph now has {self.graph.number_of_nodes()} entity nodes")

    def add_relations(self, relations: list[dict]) -> None:
        """Add edges for extracted relations, merging duplicate (subj, pred, obj) triples.

        Args:
            relations: List of relation dicts with keys ``subject_text``,
                ``predicate``, ``object_text``, ``confidence``, ``chunk_id``.
                Subject/object text is normalized to match node IDs added by
                ``add_entities``.

        Raises:
            GraphConstructionError: If a required key is missing.
        """
        try:
            for rel in relations:
                subj = _normalize(rel["subject_text"])
                obj = _normalize(rel["object_text"])
                if not subj or not obj or subj == obj:
                    continue

                # Ensure endpoint nodes exist even if add_entities wasn't called for them.
                for node_id, raw_text in ((subj, rel["subject_text"]), (obj, rel["object_text"])):
                    if not self.graph.has_node(node_id):
                        self.graph.add_node(
                            node_id,
                            label="ENTITY",
                            mention_count=0,
                            source_chunks=set(),
                            surface_forms={raw_text},
                        )

                existing_edge = self._find_existing_edge(subj, obj, rel["predicate"])
                if existing_edge is not None:
                    key = existing_edge
                    self.graph.edges[subj, obj, key]["weight"] += 1
                    self.graph.edges[subj, obj, key]["confidence"] = max(
                        self.graph.edges[subj, obj, key]["confidence"], rel["confidence"]
                    )
                    self.graph.edges[subj, obj, key]["source_chunks"].add(rel["chunk_id"])
                else:
                    self.graph.add_edge(
                        subj,
                        obj,
                        predicate=rel["predicate"],
                        confidence=rel["confidence"],
                        weight=1,
                        source_chunks={rel["chunk_id"]},
                    )
        except KeyError as exc:
            raise GraphConstructionError(f"Relation dict missing required key: {exc}") from exc

        logger.info(f"Graph now has {self.graph.number_of_edges()} relation edges")

    def _find_existing_edge(self, subj: str, obj: str, predicate: str) -> int | None:
        """Return the edge key if an identical (subj, predicate, obj) triple already exists."""
        if not self.graph.has_edge(subj, obj):
            return None
        for key, data in self.graph.get_edge_data(subj, obj).items():
            if data.get("predicate") == predicate:
                return key
        return None

    def get_neighbors(self, node_id: str, hops: int = 1) -> set[str]:
        """Return the set of node IDs reachable within N hops of a given node.

        Used by graph retrieval to expand a seed entity into its local
        neighborhood (e.g. expanding "trastuzumab" to co-occurring drugs,
        indications, and adverse effects).

        Args:
            node_id: Starting node (normalized entity text).
            hops: Number of hops to expand (bounded to keep queries fast).

        Returns:
            Set of neighboring node IDs (excludes the seed node itself).
        """
        normalized = _normalize(node_id)
        if not self.graph.has_node(normalized):
            return set()

        undirected = self.graph.to_undirected(as_view=True)
        neighbors: set[str] = {normalized}
        frontier = {normalized}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(undirected.neighbors(node))
            neighbors.update(next_frontier)
            frontier = next_frontier

        neighbors.discard(normalized)
        return neighbors

    def stats(self) -> dict:
        """Return summary statistics about the graph (for logging/reporting)."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": (
                sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0
                else 0.0
            ),
        }


def normalize_entity_text(text: str) -> str:
    """Normalize entity text for consistent node identity (lowercase, collapsed whitespace)."""
    return " ".join(text.lower().split())


# Internal alias kept so existing call sites within this module (`_normalize(...)`)
# continue to work without modification.
_normalize = normalize_entity_text
