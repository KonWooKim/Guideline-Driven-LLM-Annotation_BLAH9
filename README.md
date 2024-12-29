# Leveraging Annotation Guidelines in Prompting LLM Annotators

## Table of Contents

- [Motivation](#motivation)
- [Project Theme](#project-theme)
- [Annotation Guidelines for LLM Annotators](#annotation-guidelines-for-llm-annotators)
- [LLM Guidelines](#llm-guidelines)
- [Background](#background)
- [Dataset](#dataset)
  - [NCBI Disease Corpus](#ncbi-disease-corpus)
- [Tools](#tools)
- [Method](#method)
  - [1. Guideline Injection via Prompting](#1-guideline-injection-via-prompting)
  - [2. Knowledge-Embedding in Customized LLMs](#2-knowledge-embedding-in-customized-llms)
  - [3. Guideline-Driven Web-Based Annotation](#3-guideline-driven-web-based-annotation)
- [Evaluation](#evaluation)
  - [Quantitative Metrics](#quantitative-metrics)
  - [Qualitative Analysis](#qualitative-analysis)
- [Schedule](#schedule)
- [References](#references)

---

## Motivation

The rapid advancement of large language models (LLMs), such as GPTs, presents new opportunities to improve biomedical text annotation. However, a key challenge remains: **Can these models leverage existing guidelines to produce annotations as accurate and consistent as human-generated ones?** By integrating guideline-based domain knowledge into LLMs, we aim to enhance their performance and narrow the gap between automated and human annotation quality in the biomedical field.

---

## Project Theme

**“Optimizing Biomedical Annotations with Guideline Integration in Large Language Models”**

This project explores multiple strategies for integrating comprehensive, domain-specific annotation guidelines into LLMs, evaluating how each approach impacts annotation accuracy and consistency. By comparing results against human-annotated datasets, we will assess the effectiveness of explicitly providing or embedding domain guidelines into the model’s reasoning process.

We aim to

- Provide explicit instructions to the model using domain-specific annotation rules.
- Compare the performance of LLM-generated annotations against human annotations.
- Assess improvements in **F1-score**, **precision**, and **recall** when guidelines are incorporated.

---

## Background

Accurate biomedical annotations are critical for tasks such as disease-gene association extraction, drug efficacy studies, and systematic literature reviews. Traditionally, these tasks rely on labor-intensive manual annotations guided by detailed instructions (e.g., specifying how to tag gene or protein mentions, disease terms, drug names, etc.). While LLMs promise automation and scalability, they often fall short without explicit access to these domain-specific guidelines.

For example, **PubAnnotation** provides curated biomedical text with annotations for entities like genes, diseases, and proteins. Meanwhile, guidelines or other institution-specific protocols offer standardized rules for consistent labeling. Our hypothesis is that by explicitly incorporating these guidelines—either directly in prompts or embedded within the model—we can significantly improve LLM-driven annotation.

---

## Dataset

### NCBI Disease Corpus

![NCBI Disease Resource](./images/ncbi_disease.PNG)

- **Description**: Focuses on disease mentions in PubMed abstracts.
- **Annotations**: Disease entity annotations with normalization to Medical Subject Headings (MeSH) or Online Mendelian Inheritance in Man (OMIM).
- **Guidelines**: The corpus was annotated using a set of guidelines that specify how to identify and label disease mentions, including abbreviations and synonyms.
- **Reference**: Dogan RI, Leaman R, Lu Z. “NCBI disease corpus: A resource for disease name recognition and concept normalization.” _Journal of Biomedical Informatics_, 2014.
- **Link**: [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) or [NCBI website](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/).

---

## Tools

- **PubAnnotation**
- **Biomedical annotation guidelines** from the NCBI Disease Corpus
- **LLMs** (GPT-4 and custom GPTs)

---

## Method

We will explore three complementary approaches to guideline integration:

### 1. Guideline Injection via Prompting

**Description**  
Provide the LLM with relevant sections of the annotation guidelines as part of the prompt. For instance, summarize rules for identifying gene/protein mentions and disease terms so they fit within token limits.

**Implementation**

1. Select or summarize the most critical parts of the guideline.
2. Prompt GPT-4 (or a similar model) with these rules before annotation.

**Evaluation**  
Compare generated annotations to human annotations in PubAnnotation. Assess improvements in **F1-score**, **precision**, and **recall**.

---

### 2. Knowledge-Embedding in Customized LLMs

**Description**  
Integrate the annotation guidelines directly into knowledge in a customized GPT.

**Implementation**

1. Embed knowledge, guidelines, or biomedical text, ensuring they become part of the model’s internal knowledge.
2. Test whether the model can generalize to new texts without being explicitly prompted with the guidelines.

**Evaluation**  
Compare performance (**F1**, **precision**, **recall**) with the prompt-only method. Examine whether this approach yields higher consistency and fewer guideline violations, especially under zero-shot or few-shot conditions.

---

### 3. Guideline-Driven Web-Based Annotation

**Description**  
Develop a webpage that hosts the annotation guidelines, allowing the LLM to query them dynamically through a web search interface.

**Implementation**

1. Create a website that clearly structures the guidelines (e.g., definitions, examples, annotation cases).
2. Configure the LLM to reference the website as needed.

**Evaluation**  
Compare the annotation accuracy of dynamic referencing vs. static prompts or embedded knowledge. Investigate if real-time guideline updates improve adaptability and performance.

---

## Evaluation

### Quantitative Metrics

- **F1-Score**, **Precision**, **Recall**: Compare generated annotations to PubAnnotation’s ground truth.
- **Consistency**: Evaluate annotation uniformity across similar text segments.
- **Class-Level Analysis**: Break down performance for different entity types (e.g., genes, diseases) to identify strengths and weaknesses.

### Qualitative Analysis

- **Error Case Study**: Identify where the model fails to follow guidelines (e.g., incorrectly tagging partial entity mentions, overlooking synonyms).
- **Root Cause Analysis**: Investigate whether errors stem from unclear guidelines, model limitations, or token constraints.

---

## Schedule

| Day   | Task                                                                                                                                                           |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Day 1 | - Finalize dataset selection and gather concrete annotation guidelines<br>- Identify key entity types to focus on<br>- Test Method 1 ~ 3 and choose one        |
| Day 2 | **Symposium**                                                                                                                                                  |
| Day 3 | **Develop Further**<br>- Explore how guidelines made for human annotators can affect LLM annotators<br>- Investigate ways to narrow the gap between them       |
| Day 4 | - Conduct comparative analysis of all three methods<br>- Evaluate results against human-annotated benchmarks<br>- Summarize findings and present final results |
| Day 5 | **Final Presentation** (available until afternoon)                                                                                                             |

---

By implementing these improvements, we aim to produce an automated system that can annotate as effectively as human experts in the domain-specific area. This approach will demonstrate whether explicitly providing or embedding guidelines in large language models can significantly boost the quality and consistency of biomedical annotations—an essential step toward more efficient biomedical research and application.

---

## References

- Dogan RI, Leaman R, Lu Z. “NCBI disease corpus: A resource for disease name recognition and concept normalization.” _Journal of Biomedical Informatics_, 2014.
- [NCBI Disease Corpus (GitHub)](https://github.com/NCBI-disease-corpus)
