"""Stage 6.5: Context Compression.

Takes the top-K reranked chunks and passes them through the LLM to extract
only the facts relevant to the query. This removes noise and irrelevant
paragraphs before the final generation step, substantially reducing hallucination.

To avoid extreme latency (running the LLM N times per query), this combines
the top chunks into a single prompt and asks the LLM to extract relevant facts.
"""

from __future__ import annotations
from medgraphrag.generation.llm_generator import OllamaGenerator

class ContextCompressor:
    def __init__(self, generator: OllamaGenerator):
        self.generator = generator

    def compress(self, question: str, chunks: list[dict], text_key: str = "text") -> list[dict]:
        if not chunks:
            return []

        combined_text = "\n\n".join([f"Chunk {i+1}: {c.get(text_key, '')}" for i, c in enumerate(chunks)])
        
        system_prompt = (
            "You are a medical data extraction assistant.\n"
            "Your job is to read the provided context chunks and extract ONLY the facts, numbers, and statements that are directly relevant to answering the user's question.\n"
            "Do NOT answer the question. Just summarize the relevant facts from the chunks.\n"
            "If none of the chunks contain relevant information, reply with 'No relevant information found.'\n"
            "Keep it as concise as possible."
        )
        
        user_prompt = f"Context Chunks:\n{combined_text}\n\nQuestion: {question}\n\nExtracted Facts:"
        
        try:
            compressed_text = self.generator.generate(system_prompt, user_prompt)
        except Exception as e:
            # Fallback to original text if LLM call fails
            print(f"Context compression failed: {e}")
            return chunks

        if "No relevant information found" in compressed_text:
            # If it failed to extract, return original chunks to be safe
            return chunks
            
        # Return as a single mega-chunk to feed into the final generator
        return [{"chunk_id": "compressed_01", text_key: compressed_text, "metadata": {}}]
