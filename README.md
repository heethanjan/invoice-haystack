# Invoice Haystack

Official repository for **Invoice Haystack: Benchmarking Document Retrieval and Visual Question Answering Under Strong Visual Homogeneity**.

Invoice Haystack is a benchmark for evaluating document retrieval and visual question answering in collections of highly similar invoice documents. Unlike general multi-document benchmarks, this benchmark focuses on enterprise-style invoice repositories where documents share strong visual templates but differ in fine-grained semantic content.

## Overview

Vision-language models perform well on single-document VQA, but their performance drops when they must retrieve the correct document from large visually homogeneous collections. Invoice Haystack is designed to test this setting.

The benchmark contains:

- 1,500 anonymized invoice images
- 200 validated question-answer pairs
- Three retrieval corpus scales: 500, 1,000, and 1,500 documents
- Strong visual homogeneity, with mean pairwise cosine similarity of 0.73

## Method: VL-RAG

We also release VL-RAG, a hybrid retrieval-augmented generation framework that combines:

- OCR-based text extraction
- Dense text embeddings using BGE-Large
- Visual embeddings using SigLIP and OpenCLIP
- Average score fusion
- VLM-based binary verification filter

The retrieval pipeline ranks candidate invoices using both visual and textual signals, then applies a VLM filter to verify whether each candidate can answer the query.



## Dataset Access

The Invoice Haystack benchmark dataset is available upon request for research and evaluation purposes.

To request access, please complete the dataset request form:

**[Invoice Haystack Dataset](https://huggingface.co/datasets/heetha/invoice-haystack)**

After submitting the form, we will review the request and provide access instructions to approved users.

