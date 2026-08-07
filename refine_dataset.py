import json
import os
from tqdm import tqdm
from omegaconf import OmegaConf
from generator.llm_generator import OllamaGenerator

SYSTEM_PROMPT = """You are a strict Subtractive Editor responsible for cleaning a Question-Answer dataset used to evaluate a Medical GraphRAG system for publication in an IEEE/Scopus journal.

Your objective is to produce a high-quality benchmark dataset suitable for evaluating Retrieval Accuracy, Precision, Recall, Faithfulness, and Groundedness.
Because these metrics rely on exact phrasing and lexical overlap with the source documents, you must NEVER reword, paraphrase, or summarize the medical facts.

You will receive a JSON array of medical question-answer pairs.

Your task is to clean ONLY the "answer" field.

STRICT RULES

1. Do NOT modify:
   - id
   - question
   - difficulty
   - category

2. DO NOT paraphrase, reword, or rewrite the clinical facts. You must preserve the exact phrasing and terminology of the original provided answer. 

3. Do not introduce ANY external medical knowledge, guidelines (e.g., NCCN, ASCO), or assumptions. The final answer must remain perfectly traceable and lexically identical to the original text segments that remain.

4. Your ONLY job is to delete the following forbidden artefacts and stitch the remaining original words together into a concise sentence (if possible):
   - LOW CONFIDENCE
   - HIGH CONFIDENCE
   - UNVERIFIED
   - "The retrieved context..."
   - "The document does not provide..."
   - "It appears..."
   - "It may be..."
   - "According to..."
   - "In some studies..."
   - references (e.g., author names, et al., years like (1999))
   - citations (e.g., [P1], [P2])
   - URLs
   - conversational introductions or conclusions
   - markdown (bolding, italics)
   - bullet lists
   - numbered lists

5. Never invent information.

6. Correct basic punctuation, spacing, and capitalization only if necessary after removing artefacts.

7. Return ONLY valid JSON.

Do NOT include:
- markdown
- explanations
- notes
- comments

The output must be a valid JSON array with exactly the same structure and number of objects as the input.
"""

def main():
    print("Loading pipeline configurations...")
    model_cfg = OmegaConf.load("configs/model.yaml")
    
    generator = OllamaGenerator(
        model_name=model_cfg.llm.model_name,
        host=model_cfg.llm.host,
        temperature=0.0, # Strictly deterministic
        max_tokens=1024,
    )
    
    input_file = "result_generated.json"
    output_file = "gold_standard_dataset.json"
    
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return
        
    with open(input_file, "r") as f:
        data = json.load(f)
        
    existing_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                prev_results = json.load(f)
                for res in prev_results:
                    existing_results[res.get("id")] = res
            print(f"Loaded {len(existing_results)} already refined answers. Resuming...")
        except:
            pass
            
    print(f"Loaded {len(data)} items to validate and refine.")
    
    refined_data = []
    
    # Process 1 item at a time to prevent LLM hallucination and json format breakage
    for item in tqdm(data, desc="Refining to Gold Standard"):
        q_id = item.get("id")
        
        if q_id in existing_results:
            refined_data.append(existing_results[q_id])
            continue
            
        input_json_str = json.dumps([item], indent=2)
        
        user_prompt = f"Here is the JSON array to refine:\n\n{input_json_str}"
        
        try:
            raw_response = generator.generate(SYSTEM_PROMPT, user_prompt)
            
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            parsed_response = json.loads(cleaned_response)
            if isinstance(parsed_response, list) and len(parsed_response) > 0:
                refined_item = parsed_response[0]
                # Safely enforce unmodifiable fields regardless of what the LLM returned
                refined_item["id"] = item.get("id")
                refined_item["question"] = item.get("question")
                if "difficulty" in item:
                    refined_item["difficulty"] = item.get("difficulty")
                if "category" in item:
                    refined_item["category"] = item.get("category")
                    
                refined_data.append(refined_item)
            else:
                print(f"\nWarning: Invalid response format for {item.get('id')}")
                refined_data.append(item) 
                
        except Exception as e:
            print(f"\nError refining {item.get('id')}: {e}")
            refined_data.append(item)
            
        with open(output_file, "w") as f:
            json.dump(refined_data, f, indent=2)

    print(f"Successfully generated gold standard dataset at {output_file}")

if __name__ == "__main__":
    main()
