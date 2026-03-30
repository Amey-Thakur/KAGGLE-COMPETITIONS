<a name="readme-top"></a>
# Harmonizing the Data of your Data: SDRF Metadata Extraction Transformation
### A predictive modeling approach using high-precision rule-based extraction and ontology normalization to structure scientific proteomics metadata.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/sdrf-metadata-extraction-baseline)

---

# Hello fellow Kagglers!

This notebook presents a **high-precision rule-based pipeline** for extracting structured SDRF metadata from scientific proteomics publications. The objective is to map unstructured publication text into standardized SDRF schema fields while maximizing alignment with ground truth annotations.

The solution is built on a **tiered resolution hierarchy**:

1. training SDRF ground truth  
2. PRIDE / ProteomeXchange API retrieval  
3. rule-based text extraction  
4. heuristic fallback strategies  

Each layer progressively trades accuracy for coverage to ensure robust metadata extraction.


---

## Understanding the Problem Setting

The task is to extract structured metadata:

$$
\text{Publication Text} \rightarrow \text{SDRF Fields}
$$

Key challenges:

- highly unstructured scientific language  
- multiple valid representations of the same entity  
- strict ontology alignment requirements  
- sparse signal for certain metadata fields  

The pipeline is designed to prioritize **precision first, then recall**.


---

## 1. Data Acquisition

The environment initializes:

- dataset paths across Kaggle and local setups  
- API timeouts for external metadata retrieval  
- submission schema parsing  

### Key Step

Training SDRF files are processed to build:

- column-wise vocabularies  
- frequency statistics  

### Why this matters

Training SDRFs act as **statistical priors**, guiding fallback decisions when extraction fails.


---

## 2. Data Inspection

Statistical profiling is performed on training SDRFs.

### Components

- column frequency counters  
- vocabulary sets per metadata field  
- global mode per column  
- non-null coverage ratios  

From page 3:

- 103 SDRF files indexed  
- 48 columns exceed 50 percent coverage  

### Additional Structure

- per-PXD ground truth mapping for overlap detection  

### Why this matters

Defines:

- which fields are reliable  
- which fields require fallback  
- expected value distributions  


---

## 3. Data Cleaning

Text extraction is **context-aware**.

### Priority Sections

- Methods  
- Sample Preparation  
- Mass Spectrometry  
- Data Acquisition  

Non-standard sections are also included using keyword matching.

### Mechanism

Text is constructed by prioritizing high-signal sections and optionally including abstract and title.

### Why this matters

Metadata signals are concentrated in methodology sections. Filtering improves extraction precision.


---

## 4. Exploratory Data Analysis

Distribution analysis is performed on high-value metadata fields.

From page 7:

- dominant organism: *Homo sapiens*  
- frequent instruments: Orbitrap family  

### Purpose

- validate normalization dictionaries  
- confirm alignment with expected SDRF values  

### Why this matters

Ensures extracted values match training distribution patterns.


---

## 5. Feature Engineering (Normalization Layer)

Normalization dictionaries map raw text to ontology-compliant values.

### Examples

- organism → NCBI Taxonomy  
- tissue → UBERON  
- instrument → PSI-MS  
- modifications → UNIMOD  

### Additional Components

- FBS exclusion filter to remove false positives  
- label formatting for TMT, iTRAQ, SILAC  
- default modification priors based on frequency  

### Why this matters

Evaluation requires **exact string matching**, so normalization is critical for scoring.


---

## 6. Extraction Pipeline

### Core Function: `regex_extraction`

Extracts metadata using rule-based pattern matching.

---

### Entities Extracted

- organism, tissue, cell line, cell type  
- material type (inferred hierarchy)  
- digestion enzymes (trypsin, Lys-C, etc.)  
- labeling methods (TMT, SILAC, LFQ)  
- instruments (Orbitrap, timsTOF, etc.)  
- acquisition and fragmentation methods  
- disease, sex, specimen, genotype  
- chromatography and MS parameters  

---

### Example Mapping

Cleavage detection:

$$
\text{Trypsin} \rightarrow \text{AC=MS:1001251}
$$

---

### Additional Mechanisms

- multi-value truncation for stability  
- negative lookbehind to avoid false matches  
- clinical context filtering for disease extraction  

### Why this matters

Regex-based extraction provides **high precision and interpretability**, critical for structured schema mapping.


---

## 7. External Metadata Retrieval

Two APIs are used:

- PRIDE  
- ProteomeXchange XML  

### Extracted Fields

- organism  
- tissue  
- instrument  
- disease  
- quantification method  

### Why this matters

APIs provide structured metadata when text extraction fails, improving coverage.


---

## 8. Filename Token Parsing

Metadata is extracted directly from raw file names.

### Extracted Information

- fraction identifiers  
- biological replicates  
- technical replicates  
- labeling information  

### Why this matters

File naming conventions often encode experimental metadata not explicitly described in text.


---

## 9. Tiered Resolution Strategy

Final metadata is resolved using priority:

1. training SDRF match  
2. API retrieval  
3. regex extraction  
4. global mode fallback (restricted)  

### Constraint

Certain columns are excluded from fallback to avoid false positives.

### Why this matters

Ensures high-confidence predictions while maintaining coverage.


---

## Summary

The pipeline integrates:

1. training SDRF-based statistical priors  
2. context-aware text extraction  
3. ontology-aligned normalization  
4. rule-based metadata extraction  
5. API-based metadata augmentation  
6. filename-based metadata recovery  
7. tiered resolution strategy  

Each layer contributes to balancing precision and coverage in SDRF extraction.


---

## Closing Remarks

This notebook demonstrates that structured rule-based systems can achieve high precision in scientific metadata extraction when combined with ontology normalization and hierarchical fallback strategies.

Further improvements can focus on:

- hybrid models combining rules with learned representations  
- confidence scoring for extracted fields  
- improved disambiguation using context-aware embeddings  
- dynamic rule adaptation based on dataset statistics  

These directions can further improve robustness and scalability.


---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
