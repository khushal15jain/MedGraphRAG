"""Stage 5: Medical Entity Extraction.

Uses SciSpaCy's ``en_core_sci_sm`` model to extract biomedical entities
(diseases, drugs, procedures, anatomical terms, etc.) from each chunk.
The small model is used deliberately (per project hardware constraints) —
it trades some recall for a much smaller memory footprint, which matters
when this stage runs alongside embedding/reranker models later in the
same process lifetime on an 8GB machine.

The model is loaded once per ``MedicalEntityExtractor`` instance and reused
across all chunks (loading spaCy pipelines repeatedly is expensive).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import spacy
from spacy.language import Language

from medgraphrag.utils.exceptions import EntityExtractionError
from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MedicalEntity:
    """A single extracted biomedical entity mention."""

    entity_id: str
    text: str
    label: str          # spaCy entity label, e.g. "ENTITY" for scispacy generic NER
    start_char: int
    end_char: int
    chunk_id: str
    source_file: str
    normalized_text: str  # lowercased, whitespace-collapsed form used for graph merging


class MedicalEntityExtractor:
    """Wraps a SciSpaCy pipeline to extract entities from text chunks."""

    # Components not needed for NER-only extraction; disabling them keeps
    # the memory footprint down on the 8GB target machine, where this stage
    # runs alongside embedding/reranker models later in the process lifetime.
    _DISABLED_COMPONENTS = ["parser", "tagger", "attribute_ruler", "lemmatizer"]

    def __init__(self, spacy_model: str = "en_core_sci_sm") -> None:
        """Load the SciSpaCy pipeline.

        Args:
            spacy_model: Name of the installed SciSpaCy model.

        Raises:
            EntityExtractionError: If the model cannot be loaded (e.g. not installed).
        """
        self.model_name = spacy_model
        self.nlp = self._load_model(spacy_model)

    @staticmethod
    def _load_model(model_name: str) -> Language:
        """Load the spaCy/SciSpaCy pipeline with unneeded components disabled.

        Args:
            model_name: Name of the installed SciSpaCy model.

        Returns:
            The loaded spaCy ``Language`` pipeline.

        Raises:
            EntityExtractionError: If the model is not installed.
        """
        try:
            nlp = spacy.load(
                model_name,
                disable=MedicalEntityExtractor._DISABLED_COMPONENTS,
            )
            logger.info(
                f"Loaded SciSpaCy model '{model_name}' "
                f"(disabled: {MedicalEntityExtractor._DISABLED_COMPONENTS})"
            )
            return nlp
        except OSError as exc:
            raise EntityExtractionError(
                f"SciSpaCy model '{model_name}' is not installed. "
                f"Install it via the wheel URL listed in requirements.txt."
            ) from exc

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    def extract_from_chunk(self, chunk_text: str, chunk_id: str, source_file: str) -> list[MedicalEntity]:
        """Run NER over a single chunk's text.

        Args:
            chunk_text: The chunk's text content.
            chunk_id: Identifier of the source chunk (for provenance).
            source_file: Originating PDF filename.

        Returns:
            List of extracted ``MedicalEntity`` objects (may be empty).

        Raises:
            EntityExtractionError: If spaCy processing fails unexpectedly.
        """
        if not chunk_text.strip():
            return []

        try:
            doc = self.nlp(chunk_text)
        except Exception as exc:  # noqa: BLE001
            raise EntityExtractionError(f"NER failed on chunk {chunk_id}: {exc}") from exc

        entities: list[MedicalEntity] = []
        for ent in doc.ents:
            entities.append(
                MedicalEntity(
                    entity_id=str(uuid.uuid4()),
                    text=ent.text,
                    label=ent.label_ or "ENTITY",
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    normalized_text=self._normalize(ent.text),
                )
            )
        return entities

    def extract_from_chunks(
        self, chunks: list[dict[str, str]]
    ) -> list[MedicalEntity]:
        """Run NER over many chunks (batched via ``nlp.pipe`` for efficiency).

        Args:
            chunks: List of dicts each containing ``chunk_id``, ``text``,
                and ``source_file`` keys.

        Returns:
            Flat list of all extracted entities across all chunks.
        """
        texts = [c["text"] for c in chunks]
        
        all_entities: list[MedicalEntity] = []

        # batch_size kept modest to bound peak memory on 8GB RAM.
        processed=0
        for doc, chunk in zip(
             self.nlp.pipe(texts, batch_size=16), 
             chunks,
               strict=True
               ):
                    processed += 1

                    if processed % 500 == 0:
                         logger.info(f"NER Progress: {processed}/{len(chunks)}")
                    for ent in doc.ents:
                        all_entities.append(
                    MedicalEntity(
                        entity_id=str(uuid.uuid4()),
                        text=ent.text,
                        label=ent.label_ or "ENTITY",
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        chunk_id=chunk["chunk_id"],
                        source_file=chunk["source_file"],
                        normalized_text=self._normalize(ent.text),
                    )
                )

        logger.info(f"Extracted {len(all_entities)} entities from {len(chunks)} chunks")
        return all_entities