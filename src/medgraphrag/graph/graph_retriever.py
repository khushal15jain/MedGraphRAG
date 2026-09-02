"""Stage 11: Graph Retrieval.

Given a query, this module:
  1. Extracts candidate medical entities mentioned in the query (via the
     same SciSpaCy pipeline used during ingestion, for consistent
     normalization).
  2. Looks up those entities as seed nodes in the knowledge graph.
  3. Expands each seed node outward, hop by hop, to find related entities
     (e.g. a drug query surfaces linked indications, contraindications, and
     adverse effects even if those exact terms are not in the query).
  4. Maps the seed + expanded entities back to their source chunk IDs,
     which are then fused with dense/sparse retrieval results in
     ``retrieval/hybrid_retriever.py``.

This implements the "Graph RAG" and "Graph Expansion" components of the
research design.

v2 changes (fixing a real scoring bug + adding adaptive expansion):
  - The previous version called ``get_neighbors(seed, hops=self.expansion_hops)``
    ONCE and scored every returned neighbor as ``hop_distance=1``, regardless
    of whether it was actually 1 hop or N hops away. That flattens the
    ``mention_count / (1 + hop_distance)`` decay scoring -- a 3-hop tangential
    entity was scoring identically to a 1-hop directly-related one. This
    version does its own BFS, hop by hop, calling ``get_neighbors(node,
    hops=1)`` repeatedly and tracking the REAL distance each node was found
    at, so the decay formula actually decays.
  - Added adaptive expansion: if a seed entity's 1-hop neighborhood returns
    too few candidate chunks (graph coverage varies a lot by entity -- a
    common drug name has many neighbors, a rare syndrome may have almost
    none), automatically expand one more hop out rather than returning a
    thin result set, up to ``max_adaptive_hops``. This directly targets
    Recall@5 on the sparser regions of the graph without touching precision
    for entities that already have plenty of 1-hop coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

from medgraphrag.extraction.ner_extractor import MedicalEntityExtractor
from medgraphrag.graph.graph_builder import KnowledgeGraphBuilder, normalize_entity_text
from medgraphrag.utils.exceptions import GraphConstructionError
from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GraphRetrievalResult:
    """A chunk surfaced via graph traversal, with the entity path that found it."""

    chunk_id: str
    matched_entity: str
    hop_distance: int
    graph_score: float


class GraphRetriever:
    """Retrieves relevant chunks by traversing the medical knowledge graph."""

    def __init__(
        self,
        graph_builder: KnowledgeGraphBuilder,
        entity_extractor: MedicalEntityExtractor,
        expansion_hops: int = 1,
        adaptive_expansion: bool = True,
        max_adaptive_hops: int = 2,
        min_candidates_before_expand: int = 5,
    ) -> None:
        """Initialize the graph retriever.

        Args:
            graph_builder: A populated ``KnowledgeGraphBuilder`` instance.
            entity_extractor: Shared entity extractor for consistent query
                normalization against the graph's node IDs.
            expansion_hops: Default number of hops to expand from each seed
                entity when adaptive_expansion is off, or the STARTING hop
                count when it's on.
            adaptive_expansion: If True, automatically expand one hop
                further when a seed's neighborhood yields fewer than
                ``min_candidates_before_expand`` chunk candidates, up to
                ``max_adaptive_hops``. Backward-compatible default: this
                only ever ADDS candidates relative to the old fixed-hop
                behavior, never removes any.
            max_adaptive_hops: Ceiling on how far adaptive expansion will
                widen, regardless of how sparse a seed's neighborhood is.
            min_candidates_before_expand: Threshold (chunk count, not node
                count) that triggers one more hop of expansion.
        """
        self.graph_builder = graph_builder
        self.entity_extractor = entity_extractor
        self.expansion_hops = expansion_hops
        self.adaptive_expansion = adaptive_expansion
        self.max_adaptive_hops = max_adaptive_hops
        self.min_candidates_before_expand = min_candidates_before_expand

    def extract_query_entities(self, query: str) -> list[str]:
        """Extract and normalize entities mentioned in the query text.

        Public (previously a private helper) so HybridRetriever can extract
        entities ONCE and share them with both graph retrieval and BM25
        query expansion, instead of paying for two separate NER passes over
        the same query text.
        """
        entities = self.entity_extractor.extract_from_chunk(
            chunk_text=query, chunk_id="__query__", source_file="__query__"
        )
        return [e.normalized_text for e in entities if e.normalized_text]

    def _bfs_hop_distances(self, seed_norm: str, max_hops: int) -> dict[str, int]:
        """BFS outward from a seed node, returning {node_id: real_hop_distance}.

        Built using the undirected view of the graph directly to avoid the overhead
        of recreating it on every single node expansion.
        """
        visited = {seed_norm}
        frontier = {seed_norm}
        hop_distances: dict[str, int] = {}
        undirected = self.graph_builder.graph.to_undirected(as_view=True)

        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for node in frontier:
                if not self.graph_builder.graph.has_node(node):
                    continue
                # Skip hub nodes to prevent combinatorial explosion (and noise)
                if undirected.degree[node] > 100:
                    continue
                for neighbor in undirected.neighbors(node):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            for neighbor in next_frontier:
                hop_distances[neighbor] = hop
            visited |= next_frontier
            frontier = next_frontier

        return hop_distances

    def retrieve(
        self, query: str, top_k: int = 5, query_entities: list[str] | None = None
    ) -> list[GraphRetrievalResult]:
        """Retrieve chunks relevant to a query via graph traversal.

        Args:
            query: Natural-language clinical query.
            top_k: Maximum number of chunk results to return.
            query_entities: Optionally pass pre-extracted entities (from
                ``extract_query_entities``) to avoid a redundant NER pass
                when the caller (e.g. HybridRetriever) already extracted
                them for another purpose. If None, extracts internally --
                existing callers don't need to change anything.

        Returns:
            List of ``GraphRetrievalResult``, sorted by descending graph_score.

        Raises:
            GraphConstructionError: If graph lookups fail unexpectedly.
        """
        if query_entities is None:
            query_entities = self.extract_query_entities(query)

        if not query_entities:
            logger.debug("No entities detected in query; graph retrieval returns no results")
            return []

        chunk_scores: dict[str, GraphRetrievalResult] = {}

        try:
            for seed in query_entities:
                seed_norm = normalize_entity_text(seed)
                if not self.graph_builder.graph.has_node(seed_norm):
                    continue

                # Seed node itself: hop distance 0, highest weight.
                self._accumulate_chunk_scores(chunk_scores, seed_norm, seed_norm, hop_distance=0)

                current_hops = self.expansion_hops
                hop_distances = self._bfs_hop_distances(seed_norm, max_hops=current_hops)
                seed_chunk_count = self._count_new_chunks(chunk_scores, hop_distances)

                # Adaptive widening: this seed's neighborhood was thin --
                # expand one more hop at a time until candidates pick up or
                # we hit the ceiling. Never expands past max_adaptive_hops,
                # and never expands at all if adaptive_expansion is off.
                while (
                    self.adaptive_expansion
                    and seed_chunk_count < self.min_candidates_before_expand
                    and current_hops < self.max_adaptive_hops
                ):
                    current_hops += 1
                    hop_distances = self._bfs_hop_distances(seed_norm, max_hops=current_hops)
                    seed_chunk_count = self._count_new_chunks(chunk_scores, hop_distances)

                for neighbor, hop_distance in hop_distances.items():
                    self._accumulate_chunk_scores(
                        chunk_scores, neighbor, seed_norm, hop_distance=hop_distance
                    )
        except Exception as exc:  # noqa: BLE001
            raise GraphConstructionError(f"Graph retrieval failed for query: {exc}") from exc

        results = sorted(chunk_scores.values(), key=lambda r: r.graph_score, reverse=True)
        logger.info(
            f"Graph retrieval found {len(results)} candidate chunks from "
            f"{len(query_entities)} query entities"
        )
        return results[:top_k]

    def _count_new_chunks(
        self, existing: dict[str, GraphRetrievalResult], hop_distances: dict[str, int]
    ) -> int:
        """Rough count of how many NEW chunks this hop_distances set would add.

        Used only to decide whether adaptive expansion should widen further --
        doesn't need to be exact, just needs to distinguish "this seed found
        almost nothing" from "this seed found plenty."
        """
        count = 0
        for node_id in hop_distances:
            if not self.graph_builder.graph.has_node(node_id):
                continue
            for chunk_id in self.graph_builder.graph.nodes[node_id].get("source_chunks", set()):
                if chunk_id not in existing:
                    count += 1
        return count

    def _accumulate_chunk_scores(
        self,
        chunk_scores: dict[str, GraphRetrievalResult],
        node_id: str,
        matched_entity: str,
        hop_distance: int,
    ) -> None:
        """Update chunk_scores with chunks referenced by a given graph node.

        Score formula: score = mention_count / (1 + hop_distance), so
        directly-matched entities outrank multi-hop expansions, and
        frequently-mentioned entities (more textbook coverage) outrank rare
        ones. hop_distance is now the REAL BFS distance (see _bfs_hop_distances),
        not a flattened constant.
        """
        if not self.graph_builder.graph.has_node(node_id):
            return

        node_data = self.graph_builder.graph.nodes[node_id]
        
        # Pure topological distance decay: directly-matched seed entities score 1.0 (hop 0),
        # 1-hop neighbors score 0.5 (hop 1), 2-hop neighbors score 0.333 (hop 2).
        # Eliminates corpus mention frequency bias so common generic terms do not dominate.
        score = 1.0 / (1.0 + hop_distance)

        for chunk_id in node_data.get("source_chunks", set()):
            existing = chunk_scores.get(chunk_id)
            if existing is None or score > existing.graph_score:
                chunk_scores[chunk_id] = GraphRetrievalResult(
                    chunk_id=chunk_id,
                    matched_entity=matched_entity,
                    hop_distance=hop_distance,
                    graph_score=score,
                )
