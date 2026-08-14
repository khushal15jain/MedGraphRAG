# Reproducibility Guide & Benchmark Setup

## 1. Overview & Dataset Selection Criterion
This guide outlines the exact environment configuration, dependencies, dataset structures, and step-by-step commands required to reproduce all experimental benchmarks and statistical significance tests reported in the MedGraphRAG paper.

### Gold Standard Evaluation Dataset (N=200)
- **Dataset File**: `gold_standard_dataset.json`
- **Total Questions**: **200 Gold Clinical Oncology Questions**
- **Categories Covered**: Targeted Therapies, Resistance Mechanisms, Biomarker Profiling, Immunotherapy Guidelines, Chemotherapy Dosing, and Clinical Decision Rules.
- **Selection Criterion**: Questions were authored by clinical oncology specialists and extracted directly from 6 major oncology guidelines (NCCN, ESMO, MD Anderson Manual). All 200 questions are evaluated across all 5 pipeline configurations ($N=1000$ total evaluation inferences across the ablation sweep).

---

## 2. Sample Data for Immediate End-to-End Execution
To allow researchers to run the entire pipeline without requiring private access to the full copyrighted textbook corpus, a public-domain sample guideline is included at:
`data/raw/sample_oncology_guideline.txt`

Running `python main.py` out-of-the-box parses this sample text, extracts biomedical entities using SciSpaCy, constructs the NetworkX Knowledge Graph, indexes dense embeddings into ChromaDB, builds the inverted BM25 index, and executes test queries through local `Llama-3.2`.

---

## 3. Environment & Hardware Requirements
- **OS**: macOS (Apple Silicon M1/M2/M3/M4 recommended) or Linux (Ubuntu 22.04 LTS).
- **RAM**: Minimum 8 GB Unified Memory / System RAM.
- **Python**: 3.11.
- **Ollama Engine**: Local `Llama-3.2:latest` (4-bit quantized).

### Installation Commands
```bash
# Clone the repository
git clone git@github.com:khushal15jain/RAGupdated.git
cd RAGupdated

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Pull local Ollama LLM model
ollama pull llama3.2:latest
```

---

## 4. Reproducing All Experiments & Statistical Tests

### Step 1: Run Full End-to-End Pipeline
```bash
python main.py
```

### Step 2: Run Full 200-Question Ablation Sweep (N=1000 Inferences)
```bash
python run_ablations.py
```

### Step 3: Run Baseline Architecture Comparison Benchmark
```bash
python benchmark/run_baselines_benchmark.py
```

### Step 4: Run Statistical Hypothesis Tests (Wilcoxon Signed-Rank & Paired t-tests)
```bash
python evaluation/p_test_evaluator.py
```

### Step 5: Run Inter-Judge Agreement & Human-LLM Alignment Study
```bash
python evaluation/judge_agreement.py
```

### Step 6: Regenerate High-Resolution Publication Figures
```bash
python generate_publication_figures.py
```
*Outputs generated:* `retrieval_accuracy_chart.png`, `faithfulness_chart.png`, `groundedness_chart.png`, `hallucination_chart.png`, `clinical_reliability_chart.png`, `latency_chart.png`, `radar_chart.png`.
