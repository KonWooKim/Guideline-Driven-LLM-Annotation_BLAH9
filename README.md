# Repurposing Annotation Guidelines in Prompting LLM Annotators

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
Ultimately, an important challenge still remains: **Can LLM annotators repurpose existing guidelines to produce accurate and consistent annotations comparable to human-generated ones?**

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

Provide the LLM with the entire annotation guideline as part of the prompt. 

**Approach**

1. **Guideline Insertion**: Insert the entire guideline in the prompt.
2. **Prompt Construction**: Build a structured prompt that includes these rules alongside sample text passages.

(https://platform.openai.com/docs/guides/prompt-engineering)
(https://platform.openai.com/docs/guides/prompt-generation#prompts)

**<Details>**
  
```python
from openai import OpenAI

client = OpenAI()

META_PROMPT = """
Annotate disease mentions in a biomedically-focused text according to specific guidelines. This involves identifying specific diseases, disease classes, contiguous text strings, modifiers, duplicate mentions, minimum necessary spans, and synonymous mentions. Also, adhere to rules on entities not to annotate, such as organism names, gender, general terms, and biological processes.

# What to Annotate

## 1. Annotate all **Specific Disease** mentions
A textual string referring to a disease name may refer to a **Specific Disease** or a **Disease Class**.  
- **Disease Class**: Mentions that could be described as a family of multiple specific diseases.  
- **Specific Disease**: Mentions that can be linked to one specific definition that does not include further categorization.

**Example**  
> Diastrophic dysplasia is an autosomal recessive disease characterized by short stature, very short limbs and joint problems that restrict mobility.

- **Annotate**:  
  - “Diastrophic dysplasia” as **Specific Disease**  
  - “autosomal recessive disease” as **Disease Class**

---

## 2. Annotate **contiguous text strings**
A textual string may refer to two or more separate disease mentions. Such mentions are annotated with the **Composite Mention** category.

**Example**  
> The text phrase “Duchenne and Becker muscular dystrophy” refers to two separate diseases. If this phrase is separated into two strings: “Duchenne” and “Becker muscular dystrophy,” it results in information loss, because the word “Duchenne” on its own is not a disease mention.

---

## 3. Annotate disease mentions that are used as **Modifiers** for other concepts
A textual string may refer to a disease name, but it may modify a noun phrase (or may not be a noun phrase itself). In such cases, annotate using the **Modifier** category.

**Example**  
> Although this mutation was initially detected in four of 33 colorectal cancer families analysed from eastern England, more extensive analysis has reduced the frequency to four of 52 English HNPCC kindreds analysed.

- **Annotate**:  
  - “colorectal cancer” as **Modifier**  
  - “HNPCC” as **Modifier**

---

## 4. Annotate **duplicate mentions**
For each sentence in the PubMed abstract and title, the locations of all disease mentions are marked, including duplicates within the same sentence.

---

## 5. Annotate **minimum necessary span of text**
Use the smallest span necessary to include all tokens expressing the most specific form of the disease.

**Example**  
> In the case of “insulin-dependent diabetes mellitus,” the disease mention including the whole phrase was preferred over its substrings such as “diabetes mellitus” or “diabetes.”

---

## 6. Annotate all **synonymous mentions**
Abbreviation definitions such as “Huntington disease” (“HD”) are separated into two annotated mentions (one for the long form, one for the abbreviation).

---

# What NOT to Annotate

1. **Do not annotate organism names**  
   - Organism names such as “human” were excluded from the preferred mention.  
   - Viruses, bacteria, and other organism names were not annotated **unless** it was clear that the disease caused by these organisms is being discussed.  
   - **Example**  
     > Studies of biopsied tissue for the presence of Epstein-Barr virus and cytomegalovirus were negative.  
     
     In this case: “Epstein-Barr virus” and “cytomegalovirus” are annotated as **Specific Disease** category **only** if the context clearly indicates the diseases they cause, otherwise not.

2. **Do not annotate gender**  
   - Tokens such as “male” and “female” were only included if they specifically identified a new form of the disease.  
   - **Example**  
     > “male breast cancer”

3. **Do not annotate overlapping mentions**  
   - **Example**: The phrase “von Hippel-Lindau (VHL) disease” is annotated as **one single disease mention** (Specific Disease category).

4. **Do not annotate general terms**  
   - Very general terms such as “disease,” “syndrome,” “deficiency,” “complications,” “abnormalities,” etc. were excluded.  
   - However, the terms “cancer” and “tumor” were retained.

5. **Do not annotate references to **biological processes**  
   - e.g. Terms corresponding to biological processes such as “tumorigenesis” or “cancerogenesis.”

6. **Do not annotate disease mentions interrupted by nested mentions**  
   - Essentially, do not break the contiguous text rule.  
   - **Example**  
     > WT1 dysfunction is implicated in both neoplastic (Wilms tumor, mesothelioma, leukemia, and breast cancer) and nonneoplastic (glomerulosclerosis) disease.  
     
     Here, the mentions “neoplastic disease” and “nonneoplastic disease” are not annotated because other tokens break up these phrases.

---

# Examples

1. **“Insulin-dependent diabetes mellitus”**  
   - **Specific Disease**  
   - Prefer the whole string.

2. **CDH1 mutations predispose to early onset “colorectal cancer.”**  
   - “early onset” may or may not be part of the disease mention, depending on:  
     1. If there is a UMLS concept specifying this as a separate form of disease  
     2. If the annotator believes it should be included

3. **Human “X-linked recessive disorder”**

4. **Her fresh serum lacked complement-mediated bactericidal activity against Neisseria gonorrhoeae.**

5. **“Huntington disease” (“HD”)**  
   - The long form and the short form constitute two separate mentions.

6. **C7 deficiency**  
   - “This report represents the first cases of C7 deficiency associated with infectious complications.”  
   - “infectious complications” is too general a term; C7 deficiency is a **Specific Disease**.

7. **“colorectal, endometrial, and ovarian cancers”**  
   - Considered one **Composite Mention** of several Specific Diseases.

8. **WT1 dysfunction is implicated in both neoplastic (“Wilms tumor,” “mesothelioma,” “leukemias,” and “breast cancer”) and nonneoplastic (“glomerulosclerosis”) disease.**  
   - Potential disease mentions include: “neoplastic disease,” “nonneoplastic disease,” “Wilms tumor,” “mesothelioma,” “leukemias,” “breast cancer,” and “glomerulosclerosis.”  
   - “neoplastic disease” and “nonneoplastic disease” are **not** annotated because these phrases are interrupted by nested mentions.

9. **Tumorigenesis and cancerogenesis**  
   - These refer to processes, not diseases.

10. **“autosomal recessive disease”**  
    - Refers to a family of diseases that can be passed down genetically → **Disease Class** category.

11. **Large phenotypic variability with “convulsive disorders,” “motor retardation,” and “mental retardation.”**  
    - These are “grey area” terms: they may not correspond to a single **Specific Disease**, so they are annotated as **Disease Class** category if appropriate.

12. **Borjeson-Forssman-Lehmann syndrome (BFLS) is a syndromal X-linked mental retardation.**  
    - “Borjeson-Forssman-Lehmann syndrome” = **Specific Disease**  
    - “X-linked mental retardation syndrome” = a family of diseases (i.e., **Disease Class**)

13. **Acute meningococcal pericarditis**  
    - Exists as a separate concept in UMLS.

14. **Acute Neisseria infection**  
    - May or may not include “acute” depending on context.

15. **Classical galactosemia**  
    - Includes “classical” because it corresponds to a particular form of the disease.

16. **Inherited spinocerebellar ataxia**  
    - May or may not include “inherited,” depending on the annotator.

17. **“Adenomatous polyps of the colon and rectum”**  
    - **Composite Mention**

18. **Fibroepithelial or epithelial hyperplasias**  
    - **Composite Mention**

19. **Stage II or stage III colorectal cancer**  
    - **Composite Mention**

# Output Format

{
  "text": "string",
  "sourcedb": "string",
  "sourceid": "integer",
  "denotations": [
    {
      "id": "string",
      "span": { "begin": "integer", "end": "integer" },
      "obj": "string"
    }
  ],
   "attributes":[
      {"id":"string","subj":"string","pred":"string","obj":"string"}
   ]
}

# Examples

## Example 1
Input: "Diastrophic dysplasia is an autosomal recessive disease characterized by short stature, very short limbs, and joint problems that restrict mobility."
- Annotation:
  - "Diastrophic dysplasia": Specific Disease
  - "autosomal recessive disease": Disease Class

## Example 2
Input: "The text phrase 'Duchenne and Becker muscular dystrophy' refers to two separate diseases."
- Annotation:
  - "Duchenne and Becker muscular dystrophy": Composite Mention

## Example 3
Input: "Although this mutation was initially detected in four of 33 colorectal cancer families analysed from eastern England."
- Annotation:
  - "colorectal cancer": Modifier

# Notes

- Pay attention to context indicating whether partial terms (e.g., "acute") should be included in the annotation.
- When synonyms or abbreviations of diseases are mentioned, annotate each form separately.
- Examples provided in the document should be used as a reference framework for annotating similar mentions.
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

**</Details>**


---



### 2. Knowledge-Embedding in Customized LLMs (GPTs)

**Description**  

Integrate the annotation guidelines directly into knowledge in a customized GPT so that it inherently "learns" and "knows" the annotation rules.

**Approach**

1. **Guideline Preparation**: Prepare guidelines in any formats, such as PDF, JSON, or TXT.
2. **Knowledge Embedding**: Embed knowledge, guidelines, or biomedical text to ensure that they become part of LLM annotators' internal knowledge.

(https://openai.com/index/introducing-gpts/)

**<Details>**

![GPTs](https://github.com/user-attachments/assets/941a0f71-4dc0-4e8d-a43b-88e6687cf23e)

**</Details>**

---

### 3. Guideline-Driven Web-Based Annotation

**Description**  

Provide a live reference interface where the LLM can dynamically query guidelines on demand during the annotation task.

**Approach**

1. **Reference creation**: Create a website that clearly structures the guidelines (e.g., definitions, examples, what-to-annotate and what-not-to-annotate, and annotation cases).
2. **Configuration**: Configure the LLM to reference the website as needed.

(https://github.com/KonWooKim/Guideline)

**<Details>**

![image](https://github.com/user-attachments/assets/34494ccd-ac3d-4aef-b6fe-d11155b08dab)

**</Details>**

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

## Participants

The following individuals have contributed to this project:

- **Kon Woo Kim**
- **Rezarta Islamaj Dogan**
- **Jin-Dong Kim**

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
