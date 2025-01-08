# Leveraging Annotation Guidelines in Prompting LLM Annotators

## Table of Contents

- [Motivation](#motivation)
- [Project Theme](#project-theme)
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

Annotation often not only requires a considerable amount of time but also incurs substantial costs. Certain domains rely heavily on annotated data and seek high-quality datasets. For instance, curated biomedical data is used to advance AI and ML models, support clinical decision-making, and improve interoperability and standardization across various institutions and systems. 
The emergence of large language models (LLMs) has introduced a new paradigm to annotation tasks. Furthermore, recent studies focus on developing LLM annotators to substitute human labeling, which requires huge expenses. 

LLM annotators have achieved certain milestones in accomplishing annotation tasks, but they are still far from reaching human-level performance. One of the biggest reasons they fail is that they cannot follow the guidelines that ensure desirable outcomes. 
Ultimately, an important challenge still remains: **Can LLM annotators leverage existing guidelines to produce accurate and consistent annotations comparable to human-generated ones?**

---

## Project Theme

**“Assessing the effectiveness of guideline provision and knowledge embedding in LLM annotators”**

To integrate comprehensive and domain-specific annotation guidelines for higher accuracy and consistency, this project will begin by testing three strategies to guide (prompt) LLM annotators. After an initial comparison of these strategies, we will select the most effective one and further develop it to evaluate whether LLM annotators can achieve desirable performance compared to human annotators. By doing so, we will assess the effectiveness of guideline provision and knowledge embedding in LLM annotators.

We aim to:

- Provide explicit instructions to LLM annotators using existing domain-specific annotation guidelines.
- Compare the performance of LLM-generated annotations against the gold-standard annotations made by human annotators.
- Assess improvements in **F1-score**, **precision**, and **recall** when guidelines are incorporated.
- Investigate the performance gap between LLM annotators and human annotators.
- Explore further strategies to improve the understandability of LLM annotators and narrow the gap.

---

## Background

Curated annotations are crucial in the biomedical field because they are closely related to clinical decision-making, which is highly associated with human health. Traditionally, annotation tasks heavily depend on labor-intensive activities guided by detailed instructions. Although LLM annotators promise automation and scalability, they still fall short of achieving human-level performance for various reasons.

One of the key obstacles is that LLM annotators do not fully understand what data stakeholders want to collect because they lack sufficient knowledge and are not guided or trained in the same way human annotators are. Our hypothesis is that **LLM-driven annotations could be significantly improved if detailed and explicit guidelines are provided and sufficient knowledge and instructions are embedded through the prompts generated for LLM annotators.**


---

## Dataset

### NCBI Disease Corpus

<!-- ![NCBI Disease Resource](./images/ncbi_disease.PNG) -->

- **Description**: Focuses on disease mentions in PubMed abstracts.
- **Annotations**: Disease entity linked to Medical Subject Headings (MeSH) or Online Mendelian Inheritance in Man (OMIM).
- **Guidelines**: The corpus was annotated using a set of guidelines that specify how to identify and label disease mentions, including abbreviations and synonyms.
- **Reference**: Dogan RI, Leaman R, Lu Z. “NCBI disease corpus: A resource for disease name recognition and concept normalization.” _Journal of Biomedical Informatics_, 2014.
- **Link**: [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) or [NCBI website](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/).

---

## Tools

- [**PubAnnotation**](https://www.pubannotation.org/) - Open web-based platform designed to share and manage text annotations
- [**Biomedical annotation guidelines**](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/Guidelines.html) - from the NCBI Disease Corpus
- [**LLMs**](https://platform.openai.com/docs/models/gpt-4o) - GPT-4 and custom GPTs

---

## Method

We will test three compatible approaches to knowledge embedding with guideline provision:

### 1. Guideline Injection via Prompting

**Description**  

Provide the LLM with relevant sections of the annotation guidelines as part of the prompt. For instance, summarize rules for identifying gene/protein mentions and disease terms so they fit within token limits.
Focus on disease, gene/protein, and related biomedical entities relevant to the NCBI Disease Corpus.

**Implementation**

1. **Guideline Selection**: Extract the most essential rules from the official guidelines (e.g., how to handle abbreviations or synonyms).
2. **Prompt Construction**: Build a structured prompt that includes these rules alongside sample text passages.
3. **LLM Annotation**: Prompt GPT-4 (or similar models) to label relevant entities.
4. **Evaluation**: Compare the generated annotations with the gold-standard annotations using PubAnnotation.

(https://platform.openai.com/docs/guides/prompt-engineering)
(https://platform.openai.com/docs/guides/prompt-generation#prompts)
  
```python
from openai import OpenAI

client = OpenAI()

META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()

def generate_prompt(task_or_prompt: str):
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {
              "role": "system",
              "content": META_PROMPT,
          },
          {
              "role": "user",
              "content": "Task, Goal, or Current Prompt:\n" + task_or_prompt,
          },
      ],
  )

  return completion.choices[0].message.content
```

---

### 2. Knowledge-Embedding in Customized LLMs (GPTs)

**Description**  

Integrate the annotation guidelines directly into knowledge in a customized GPT so that it inherently "learns" and "knows" the annotation rules.

**Implementation**

1. **Guideline Preparation**: Prepare guidelines in any formats, such as PDF, JSON, or TXT.
2. **Knowledge Embedding**: Embed knowledge, guidelines, or biomedical text to ensure that they become part of LLM annotators' internal knowledge.
3. **LLM Annotations**: ask LLM annotators to label relevant entities.
4. **Evaluation**: Compare the generated annotations with the gold-standard annotations using PubAnnotation.

(https://openai.com/index/introducing-gpts/)

![GPTs](https://github.com/user-attachments/assets/941a0f71-4dc0-4e8d-a43b-88e6687cf23e)

---

### 3. Guideline-Driven Web-Based Annotation

**Description**  

Provide a live reference interface where the LLM can dynamically query guidelines on demand during the annotation task.

**Implementation**

1. **Reference creation**: Create a website that clearly structures the guidelines (e.g., definitions, examples, what-to-annotate and what-not-to-annotate, and annotation cases).
2. **Configuration**: Configure the LLM to reference the website as needed.
3. **LLM Annotations**: ask LLM annotators to label relevant entities.
4. **Evaluation**: Compare the generated annotations with the gold-standard annotations using PubAnnotation.

(https://github.com/KonWooKim/Guideline)

![image](https://github.com/user-attachments/assets/34494ccd-ac3d-4aef-b6fe-d11155b08dab)

---

## Evaluation

### Quantitative Metrics

- **F1-Score**, **Precision**, **Recall**: Compare the generated annotations to the ground truth using PubAnnotation.
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
| Day 4 | - Conduct comparative analysis of all three methods<br>- Evaluate results against the gold-standard annotations<br>- Summarize findings and present final results |
| Day 5 | **Final Presentation** (available until afternoon)                                                                                                             |

---

### Extensions and Future Work

- **Multi-Guideline Integration**: Explore how multiple sets of overlapping or conflicting guidelines can be harmonized within the same LLM.
- **Cross-Domain Transfer**: Investigate whether guidelines for one biomedical subfield (e.g., disease annotation) can generalize to others (e.g., drug or gene annotation).
- **Active Learning Approaches**: Combine human-in-the-loop revisions with automated annotation to continually refine guideline interpretations and model performance.
- **Application to Other Biomedical Datasets**: Beyond NCBI Disease Corpus, test on resources like BC5CDR (BioCreative V CDR Task) to confirm generalizability.

---

Through this project, we aim to develop LLM annotators that can achieve annotation performance comparable to that of human experts in the biomedical field. Our approach will demonstrate whether knowledge embedding, combined with guideline provision, can improve the quality and consistency of complex annotation tasks — **an essential step toward more time- and cost-efficient biomedical research and applications.**

---

## References

- Dogan RI, Leaman R, Lu Z. “NCBI disease corpus: A resource for disease name recognition and concept normalization.” _Journal of Biomedical Informatics_, 2014.
- [NCBI Disease Corpus (GitHub)](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)
