# MedGraphRAG Source Corpus Documentation

This document describes the medical oncology reference corpus supported by the MedGraphRAG architecture.

> **IMPORTANT REPRODUCIBILITY NOTICE**  
> Copyrighted medical textbooks and clinical guidelines are **not** publicly distributed with this open-source repository. To perform full end-to-end PDF ingestion, users must independently obtain the source documents listed below and place PDF files in `data/raw/`. For out-of-the-box pipeline demonstration without copyrighted text, a synthetic guideline is provided in `data/raw/sample_oncology_guideline.txt`.

---

## Source Reference Documents

| Document ID | Source Title | Publisher / Organization | Edition / Version | Used in Evaluation | Access & Licensing |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **CORPUS-01** | NCCN Clinical Practice Guidelines in Oncology | National Comprehensive Cancer Network (NCCN) | v.3.2024 | Yes | Educational/Clinical Access via [NCCN.org](https://www.nccn.org) |
| **CORPUS-02** | ESMO Handbook of Immuno-Oncology | European Society for Medical Oncology (ESMO) | 2nd Edition (2023) | Yes | Member Access via [ESMO.org](https://www.esmo.org) |
| **CORPUS-03** | MD Anderson Manual of Medical Oncology | McGraw-Hill Education | 3rd Edition (2017) | Yes | Commercial Publication |
| **CORPUS-04** | Oxford Handbook of Oncology | Oxford University Press | 4th Edition (2015) | Yes | Commercial Publication |
| **CORPUS-05** | Cavalli Textbook of Medical Oncology | CRC Press | 4th Edition (2009) | Evaluation Corpus | Commercial Publication |
| **CORPUS-06** | Cancer: Principles & Practice of Oncology | Wolters Kluwer / Lippincott Williams & Wilkins | 6th Edition (2001) | Evaluation Corpus | Commercial Publication |
| **CORPUS-DEMO** | Sample Medical Oncology Clinical Guideline | MedGraphRAG Repository | Open-Source Synthetic | Demo Pipeline | MIT License (`data/raw/sample_oncology_guideline.txt`) |

---

## Document Ingestion Instructions

1. Obtain licensed PDF copies of NCCN or ESMO oncology guidelines.
2. Save raw PDF files into `data/raw/`.
3. Execute document parsing, chunking, and Knowledge Graph indexing:
   ```bash
   python main.py
   ```
