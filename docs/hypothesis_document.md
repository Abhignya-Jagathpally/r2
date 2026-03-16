# Hypothesis Document: Pathway-Level Latent Variables for MM Risk Prediction Across Microarray & RNA-seq

## Core Hypothesis

**"Pathway-level latent variables (derived from ssGSEA, GSVA, or topic modeling of biological pathways) transfer with better generalization across microarray→RNA-seq platforms for Multiple Myeloma risk prediction than raw gene-level signatures, with expected cross-platform C-index drop <5% for pathway-level vs. >10% for gene-level models."**

### Biological Rationale

1. **Gene-level signal is platform-specific:** Microarray measures fluorescence intensity; RNA-seq measures digital counts. Gene-by-gene differences in technical variability are large and platform-dependent. A gene highly robust on microarray (low CV) may be noisy on RNA-seq (high CV across technical replicates).

2. **Pathway-level abstraction buffers platform noise:** Biological pathways (e.g., "MAPK signaling," "immune suppression," "cell cycle") aggregate 50+ genes with coordinated function. Within a pathway, platform-specific measurement noise on individual genes should cancel out via averaging/aggregation. ssGSEA (rank-based) and GSVA (kernel density) are inherently more robust to outlier genes than mean expression.

3. **Existing signatures optimized for microarray:** GEP70 and SKY92 were discovered and validated on Affymetrix U133Plus2.0 microarray. The 70/92-gene lists have implicit platform bias (gene selection favored robust microarray signals). Transitioning to RNA-seq without re-optimization will incur performance loss.

4. **Pathway-level models reduce overfitting:** Microarray cohorts used for GEP70/SKY92 discovery had limited sample sizes (n=280 discovery, n=500-1000 validation). With 70 genes, overfitting risk is high. Pathway-level models (10-20 pathways, not 70 genes) have fewer parameters, lower overfitting risk, and better expected generalization.

---

## Prior Evidence For the Hypothesis

### Direct Evidence (Strong)

1. **Tarazona & Furio-Tari, 2024 (PLOS Computational Biology)**
   - "Construct prognostic models of multiple myeloma with pathway information incorporated"
   - **Finding:** ssGSEA and GSVA achieved comparable or superior accuracy to gene-level models in MM within the same cohort (microarray)
   - **C-indices:** Gene model ~0.72, ssGSEA on Vax pathways ~0.72, GSVA on immune pathways ~0.71
   - **Key advantage:** Pathway models used 12-20 features vs. 50+ genes; interpretability superior without sacrificing accuracy
   - **Limitation:** Did not test cross-platform (no RNA-seq validation)

2. **Shao et al., 2025 (Scientific Reports, Cancer Research)**
   - "Domain Adaptation Enhances Cross-Cohort Generalization in Breast Cancer Gene Expression Prediction"
   - **Finding:** Domain adaptation + latent variable models showed <5% C-index drop when transferring breast cancer model from TCGA-BRCA to METABRIC (different platforms, different batches)
   - **Mechanism:** Graph neural networks + domain adversarial training aligned pathway-level representations; raw gene-level transfer dropped ~10-15%
   - **Implication:** If breast cancer pathway transfer works, should work in MM (similar transcriptomic principle)
   - **Limitation:** Breast cancer, not MM; biology may differ

3. **Cross-Platform Normalization Studies (Tian et al. 2023, Nature Communications)**
   - "Cross-platform normalization enables machine learning model training on microarray and RNA-seq data simultaneously"
   - **Finding:** ComBat batch correction allowed simultaneous training on microarray + RNA-seq (treating platforms as batch)
   - **Performance:** Training on both platforms generalized better than training on single platform
   - **Limitation:** Tested on gene-level features, not pathways; unclear if pathway-level transfer would be even better

### Indirect Evidence (Supporting)

4. **GSVA/ssGSEA Robustness Literature (Hänzelmann 2013, Barbie 2009)**
   - GSVA designed to be platform-agnostic (not dependent on absolute expression levels)
   - ssGSEA rank-based; inherently robust to outlier genes
   - Both methods validated across microarray + RNA-seq datasets in numerous cancer studies (breast, lung, etc.)
   - **Implication:** If GSVA/ssGSEA work in other cancers, should work in MM

5. **Single-Cell RNA-Seq Studies Show Pathway Stability Across Cell Types (MMRF 2024, Desmedt 2021)**
   - Pathway-level genes show consistent co-regulation even across different plasma cell populations (normal vs. malignant)
   - Immune pathway markers (LILRB4, PD-L1, TIM3) robustly associated with MM prognosis across bulk + single-cell data
   - **Implication:** Pathway-level biology more stable than individual gene expression

6. **Microarray-to-RNA-seq Gene-Level Concordance Studies**
   - Correlation of top differentially expressed genes between microarray + RNA-seq typically r=0.70-0.80 (moderate)
   - **Gap:** Signature-level (70-gene list) concordance not reported; likely lower due to ranking changes
   - **Implication:** Gene-level transfer will lose signal; pathway-level averaging may recover it

---

## Prior Evidence Against the Hypothesis

### Counterarguments (Moderate Concern)

1. **Pathway Databases May Be Noisy in MM Context**
   - Pathways (MSigDB, KEGG, Reactome) curated primarily in epithelial cancers, not hematologic
   - MM has unique drivers (cyclin D, translocations, MAPK); standard pathway definitions may not be optimal
   - **Risk:** If pathway definitions suboptimal, pathway-level models may not improve over genes
   - **Mitigation:** Test with custom MM-derived pathways (e.g., MAPK, cyclin D, immunosuppression)

2. **Latent Variable Models May Overfit in Small MM Cohorts**
   - MM discovery cohorts (GEP70 n=280, SKY92 n=290) are small compared to modern breast cancer cohorts (n=1000+)
   - Topic modeling / LDA requires sufficient samples; overfitting risk with n<500
   - **Risk:** Pathway latent variables may not generalize better if training cohort too small
   - **Mitigation:** Validate on larger cohorts (MMRF CoMMpass, GEO aggregated)

3. **RNA-seq Tumor Purity Confounding May Dominate**
   - Even if pathway-level features more robust than genes, tumor purity (~40% normal PCs in bulk) may introduce similar noise in both platform
   - **Risk:** Platform difference secondary to purity issue
   - **Mitigation:** Validate in purified malignant vs. normal PC populations (CD138++/- sorting or flow isolation)

4. **No Direct MM Cross-Platform Transcriptomics Validation Study Exists**
   - Direct microarray→RNA-seq comparison in MM is rare; makes hypothesis untestable with current literature alone
   - **Risk:** Cross-platform concordance may be better than expected (gene transfer sufficient) or worse (pathway transfer also insufficient)
   - **Mitigation:** Conduct prospective cross-platform validation study

---

## Experimental Design

### Study Overview

**Objective:** Compare pathway-level vs. gene-level MM risk signature transfer across microarray→RNA-seq; quantify cross-platform generalization loss.

**Study Type:** Retrospective + prospective validation (Cohort 1 discovery, Cohorts 2-3 validation)

**Power:** Detect 5% difference in C-index drop between pathway-level (expected <5%) and gene-level (expected >10%) with 80% power, n=150-200 per cohort.

---

### Cohort Specifications

#### Cohort 1 (Discovery): Legacy Microarray + Modern RNA-seq Integration

**Source:** MMRF CoMMpass study (MMRF is largest MM genomic repository, n=1000+ NDMM)
- **Microarray subset:** n=300-400 NDMM patients with Affymetrix U133Plus2.0 microarray (legacy HOVON-65, GMMG-HD4 archives)
- **RNA-seq subset:** n=300-400 NDMM patients from MMRF RNA-seq cohort (Illumina HiSeq)
- **Matching criteria:** Stratified by risk (ISS, FISH cytogenetics) to ensure similar risk distribution
- **Clinical outcomes:** PFS, OS, treatment type (alk transplant-eligible, non-eligible), follow-up ≥3 years

**Sample requirements per patient:**
- Baseline bone marrow CD138+ plasma cells (purified, ideally >70% tumor cells; purity metadata required)
- Molecular data: Microarray (U133Plus2.0) OR RNA-seq (Illumina, raw counts + normalized data)
- Cytogenetics: FISH panel (del17p, t4;14, +1q, del1p32)
- Clinical: Baseline ISS, LDH, β2-microglobulin, treatment, outcome

---

#### Cohort 2 (External Validation #1): RNA-seq Independent Cohort

**Source:** MMRF CoMMpass RNA-seq cohort (n=300-400, independent from Discovery Cohort 1)
- **Platform:** RNA-seq only (no microarray)
- **Stratification:** Ensure similar baseline risk to Cohort 1

**Rationale:** Validate that pathway models trained on Cohort 1 generalize to independent RNA-seq data

---

#### Cohort 3 (External Validation #2): Microarray-Only Validation

**Source:** Publicly available GEO/ArrayExpress MM cohorts (e.g., GSE9782, GSE24080)
- **Platform:** Affymetrix U133Plus2.0 (legacy microarray)
- **Sample size:** n=100-150 (smaller public datasets)
- **Outcomes:** OS, PFS if available; otherwise, event-free survival or published risk scores

**Rationale:** Confirm that pathway models retain accuracy in microarray-only context (not degraded by attempts at cross-platform harmony)

---

### Study Phases

#### Phase 1: Data Preprocessing & Pathway Scoring

**Step 1a: Microarray Preprocessing**
- RMA normalization (standard for GEP70/SKY92)
- Batch correction across microarray batches (if multiple sites): ComBat or SVA
- QC: Exclude samples with outlier quality metrics (PCA, array-level intensity)
- Output: n_samples × n_genes expression matrix (log2)

**Step 1b: RNA-seq Preprocessing**
- Raw counts normalization: TMM (edgeR) or DESeq2 (variance-stabilizing)
- Batch correction across sequencing batches: ComBat-seq
- QC: Exclude low-depth samples (<5M mapped reads) or high contamination
- Output: n_samples × n_genes normalized counts matrix

**Step 1c: Pathway Scoring (Both Platforms)**

For each sample, calculate pathway-level scores using:

1. **ssGSEA** (rank-based, platform-agnostic)
   - Pathway databases:
     - Biological Process (MSigDB C5)
     - KEGG canonical pathways (C2:KEGG)
     - MM-custom: MAPK, cell cycle, immunosuppression, proliferation, protein synthesis (self-curated, n=10-15 pathways)
   - Output: n_samples × n_pathways score matrix (continuous, 0-1)

2. **GSVA** (kernel density estimation)
   - Same pathway databases as ssGSEA
   - Output: n_samples × n_pathways score matrix (continuous, unbounded)

3. **Topic Modeling (Latent Dirichlet Allocation, dLDA)**
   - Discretize expression into 3 bins (low, medium, high) per gene across all samples
   - Fit dLDA with K=7-10 topics (MM-specific; representing distinct biological programs)
   - Output: n_samples × K topic distribution (mixture of topics per patient)

---

#### Phase 2: Risk Model Development

**Step 2a: Gene-Level Models (Baseline)**

Train Cox proportional hazards (CPH) + Random Survival Forest (RSF) models on Cohort 1 microarray:

1. **Raw genes (70-gene signature):**
   - Feature set: GEP70 gene list (established, allows comparison to literature)
   - Outcome: PFS (primary), OS (secondary)
   - Validation: 5-fold cross-validation on Cohort 1 microarray only
   - **Report:** C-index, calibration plot, HR for high vs. low risk

2. **Alternative raw genes (SMM of 30 prognostic genes):**
   - Feature set: Univariate Cox-selected top 30 genes (p<0.05) from Cohort 1 microarray
   - Outcome: PFS
   - **Report:** C-index (expected lower due to overfitting, illustrating gene-level problem)

**Step 2b: Pathway-Level Models (Main Hypothesis)**

Train same CPH + RSF models on pathway-level features (ssGSEA, GSVA, dLDA topics):

1. **ssGSEA-based risk model:**
   - Feature set: ssGSEA scores for 12-15 key MM pathways (MAPK, cell cycle, immune suppression, etc.)
   - Outcome: PFS
   - Validation: 5-fold cross-validation on Cohort 1 microarray
   - **Report:** C-index, feature importance

2. **GSVA-based risk model:**
   - Feature set: GSVA scores (same pathways)
   - **Report:** C-index (expect similar to ssGSEA)

3. **Topic modeling-based risk model (dLDA):**
   - Feature set: 7-10 latent topic distributions per sample
   - Method: Supervise topics using outcome (PFS) via MTLR or Cox regression on topics
   - **Report:** C-index, interpretability of topics (which biological programs most prognostic)

---

#### Phase 3: Cross-Platform Generalization Testing

**Step 3a: Microarray→RNA-seq Transfer (Main Test)**

1. **Train pathway-level models on Cohort 1 Microarray; Test on Cohort 1 RNA-seq:**
   - Pathway scores calculated independently for microarray + RNA-seq (using same pathway databases, same ssGSEA/GSVA parameters)
   - Trained CPH/RSF from microarray applied directly to RNA-seq pathway scores (no retraining)
   - **Outcome:** C-index_RNA-seq, compare to C-index_microarray
   - **Calculate:** ΔC-index_pathway = C-index_microarray - C-index_RNA-seq (expected <5%)

2. **Train gene-level models on Cohort 1 Microarray; Test on Cohort 1 RNA-seq:**
   - Use same GEP70 gene list; map genes to RNA-seq data
   - Apply trained CPH/RSF to RNA-seq data
   - **Outcome:** C-index_RNA-seq_genes
   - **Calculate:** ΔC-index_genes = C-index_microarray - C-index_RNA-seq_genes (expected >10%)

3. **Statistical Comparison:**
   - Test H0: ΔC-index_pathway = ΔC-index_genes (no difference in transfer loss)
   - Expected: ΔC-index_pathway < ΔC-index_genes by >5% (with 95% CI)

---

**Step 3b: RNA-seq→RNA-seq Validation (Sanity Check)**

1. **Train on Cohort 1 RNA-seq; Test on Cohort 2 RNA-seq:**
   - Same-platform transfer; should have minimal loss (ΔC-index ~2-3%)
   - **Purpose:** Confirm that pathway models work within RNA-seq context (not artificially degraded)

---

#### Phase 4: External Validation

**Step 4a: Validation in Cohort 3 (Microarray-only GEO):**

1. **Train pathway models on Cohort 1 Microarray; Apply to Cohort 3 Microarray:**
   - Confirm that pathway models generalize across microarray datasets (validation of within-platform transferability)
   - **Expected:** C-index drop <5% (similar to within-Cohort 1)

2. **Compare to published GEP70/SKY92 scores (if available):**
   - Ensure pathway-level models not inferior to existing gold-standard signatures

---

### Statistical Analysis Plan

#### Primary Outcome: Cross-Platform C-index Comparison

**Test 1: Pathway-level C-index drop is <5%**
- H0: ΔC-index_pathway ≥ 5%
- H1: ΔC-index_pathway < 5%
- Method: One-sided 95% CI for ΔC-index; reject H0 if upper limit <5%
- **Alpha:** 0.05 (one-sided)

**Test 2: Gene-level C-index drop is >10%**
- H0: ΔC-index_genes ≤ 10%
- H1: ΔC-index_genes > 10%
- Method: One-sided 95% CI for ΔC-index
- **Alpha:** 0.05 (one-sided)

**Test 3: Pathway superiority (primary test)**
- H0: ΔC-index_pathway ≥ ΔC-index_genes
- H1: ΔC-index_pathway < ΔC-index_genes (pathway transfer loss smaller)
- Method: Two-sided 95% CI for difference (ΔC-index_genes - ΔC-index_pathway); reject H0 if lower limit >0
- **Alpha:** 0.05 (two-sided)
- **Expected effect size:** ΔC-index_genes - ΔC-index_pathway ≈ 5-10% (absolute C-index difference)

#### Secondary Outcomes

**Outcome 1: Calibration Slope (Agreement Between Predicted & Observed Risk)**
- Calculate calibration slope in Cohort 1 RNA-seq; expect ≤10% change from microarray
- Pathway models expected to maintain slope better than gene models

**Outcome 2: Feature Stability (Top Pathway Features Consistent Across Platforms)**
- Calculate feature importance (mean decrease in accuracy, Gini, etc.) for each pathway
- Expect >80% of top 10 pathways same rank on microarray vs. RNA-seq for pathway models
- Gene models expected to show rank shuffling (top 10 genes may change)

**Outcome 3: Sensitivity/Specificity for Risk Classification**
- Define HR cutoff on Cohort 1 microarray; apply to RNA-seq
- Report sensitivity/specificity for HR classification (vs. FISH cytogenetics + ISS)
- Expect pathway model ≥90% sensitivity, ≥85% specificity across platforms
- Gene model expected to drop sensitivity/specificity more on RNA-seq

---

### Expected Effect Sizes

#### Primary Hypothesis

| Metric | Pathway-Level | Gene-Level | Expected Difference |
|---|---|---|---|
| **C-index (Cohort 1 Microarray)** | 0.72-0.75 | 0.70-0.73 | Pathway slightly better (~0.01-0.02) |
| **C-index (Cohort 1 RNA-seq)** | 0.69-0.72 | 0.62-0.66 | Pathway substantially better (~0.05-0.06) |
| **ΔC-index (Microarray→RNA-seq)** | 0.02-0.04 | 0.08-0.12 | **Primary test: Difference 0.05-0.10** |
| **95% CI for (ΔGenes - ΔPathways)** | (0.02, 0.12) | — | **Exclude 0 → pathway better** |

#### Secondary Hypotheses

- **Calibration slope:** Pathway drop 5-10% vs. gene drop 15-25%
- **Feature stability:** Pathway top-10 concordance 80-90% vs. gene 40-60%
- **HR sensitivity/specificity:** Pathway 90-95% / 85-90% vs. gene 80-85% / 70-80%

---

## Statistical Testing Plan

### Sample Size Justification

**Primary test:** Two-sided test for H0: ΔC-index_pathway = ΔC-index_genes

Assume:
- ΔC-index_pathway = 3% (SD 2%)
- ΔC-index_genes = 10% (SD 3%)
- Correlation between estimates r = 0.5
- Desired effect size: 7% difference, 95% power, 0.05 alpha
- **N = 150-200 per cohort** sufficient (accounting for clustering, outcome prevalence ~40% PFS events)

**Larger sample** (N=300-400) desirable for:
- Subgroup analyses (risk stratification by ISS, FISH subgroup)
- Robustness to missing data
- External validation in independent cohorts

---

### Data Analysis

**Statistical software:** R (survival, randomForestSRC, survminer, pROC, glmnet)

**Primary analyses:**
1. Concordance index (Harrell's C) with 95% CI (1000-fold bootstrap)
2. Calibration slope + intercept (logistic regression of PFS outcome on risk score)
3. Net Benefit (decision curve analysis) at clinically relevant threshold (e.g., 50% 2-year PFS risk)
4. Interaction test (platform × model type) via Cox PH with interaction term

**Sensitivity analyses:**
1. Exclude samples <70% tumor purity; refit models
2. Use alternative pathway databases (KEGG, Biocarta); compare to MSigDB
3. Stratify by treatment era (pre-IMiD, IMiD-era, bortezomib-era); test interaction
4. Reweight by outcome prevalence (if unbalanced across platforms); use weighted AUC

**Subgroup analyses:**
- By ISS stage (I, II, III)
- By FISH risk (standard vs. high-risk cytogenetics)
- By age (<65 vs. ≥65)
- By treatment type (transplant-eligible vs. ineligible)

---

### Software & Computational Requirements

- **R packages:** survival, randomForestSRC, GSVA, sva, ComBat-seq, topicmodels, glmnet
- **Pathway databases:** MSigDB (C2, C5), KEGG, custom MM pathways
- **Compute time:** ~2-4 weeks for full pipeline (preprocessing, modeling, cross-validation, validation)
- **Data storage:** ~50 GB (raw + processed expression matrices, outputs)

---

## Expected Outcomes & Interpretation

### Success Scenario (Supports Hypothesis)

**Expected findings:**
- C-index_pathway (RNA-seq) ≥ 0.69 (≥95% of microarray C-index)
- ΔC-index_pathway = 2-4% (minimal loss)
- ΔC-index_genes = 10-15% (substantial loss)
- **Difference 95% CI excludes 0:** pathway transfer loss significantly <gene transfer loss
- Feature stability: pathway top-10 conserved 80-90% across platforms

**Interpretation:**
- Pathway-level abstraction effectively buffers cross-platform batch effects
- Pathway-based risk models should be developed for RNA-seq clinical assays
- Legacy microarray GEP70/SKY92 can be converted to pathway-based versions for RNA-seq with <5% expected loss

**Clinical implication:**
- Recommend developing RNA-seq—based pathway signature assay (10-15 pathway features, platform-agnostic) for routine MM risk stratification
- Microarray-based GEP assays can transition with minimal revalidation if pathway-level harmonization used

---

### Null Scenario (Against Hypothesis)

**Expected findings:**
- ΔC-index_pathway ≈ ΔC-index_genes (~10% for both)
- Feature stability similar (both show rank shuffling)
- No interaction between pathway/gene and platform

**Interpretation:**
- Pathway abstraction does not meaningfully buffer platform effects
- Cross-platform batch effects dominate both approaches
- Alternative explanation: Platform differences reflect fundamental biological variation (e.g., tumor purity confounding), not just technical noise

**Clinical implication:**
- Focus on rigorous batch correction (ComBat-seq, SVA) for raw genes, not pathway-level approaches
- Consider microarray→RNA-seq transfer as requiring full revalidation + retraining, not accommodation

---

### Intermediate Scenario (Partial Support)

**Expected findings:**
- ΔC-index_pathway = 5-8% (moderately reduced from gene drop of 10-12%)
- Modest improvement in pathway transfer, but not >5% threshold

**Interpretation:**
- Pathway abstraction provides modest benefit; partial explanation of cross-platform robustness
- Other factors (pathway database quality, optimal # of pathways, topic model hyperparameters) may need tuning
- May be worth developing pathway-based assays, but improvement incremental rather than transformative

**Clinical implication:**
- Recommend hybrid approach: pathway-based main features (10-15) + critical gene-level surrogates (e.g., MMI detected FISH surrogates) for RNA-seq assay

---

## Deviations & Contingencies

### If Cohort Size Insufficient

- Use published GEO MM cohorts in meta-analysis approach (combining multiple small datasets)
- Reduce pathway features (5-10) to increase power per parameter
- Focus on single pathway method (ssGSEA) rather than comparing all three

### If Cross-Platform Harmonization Inadequate

- Apply more aggressive batch correction: Seurat integration (for RNA-seq) or cross-platform ComBat (for cross-omics)
- Include platform as covariate in Cox model (to isolate biological effects from technical confounding)
- Stratify analyses by batch if heterogeneous

### If Pathway Databases Suboptimal for MM

- Develop MM-specific pathway definitions from:
  - Gene co-expression networks (WGCNA) in MM RNA-seq cohorts
  - KEGG/Biocarta pathways curated for MM biology (MAPK, cell cycle, cyclin D, etc.)
  - Immunosuppression pathways from scRNA-seq immune atlas literature
- Refit pathway scores with custom databases; compare to standard MSigDB

### If Topic Modeling Does Not Improve Over ssGSEA/GSVA

- Simplify analysis: focus on ssGSEA ± GSVA (both established, faster)
- Topic modeling exploratory; not part of primary hypothesis test
- Rationale: ssGSEA/GSVA sufficient to test pathway-level vs. gene-level hypothesis

---

## Timeline & Milestones

| Phase | Milestone | Timeline |
|---|---|---|
| 1 | Data acquisition (Cohorts 1-3) | Months 0-2 |
| 2 | Preprocessing + QC | Months 2-3 |
| 3 | Pathway scoring (ssGSEA, GSVA, dLDA) | Months 3-4 |
| 4 | Risk model development (CPH, RSF) | Months 4-5 |
| 5 | Cross-platform validation | Months 5-6 |
| 6 | External validation (Cohort 3) | Months 6-7 |
| 7 | Sensitivity/subgroup analyses | Months 7-8 |
| 8 | Manuscript preparation | Months 8-9 |
| 9 | Submission + revision | Months 9-12 |

---

## References (Key Hypothesis-Informing Papers)

1. Tarazona, S., et al. (2024). "Construct prognostic models of multiple myeloma with pathway information incorporated." *PLOS Computational Biology* 20(9):e1012444. → **Pathway >gene in MM (within-platform)**

2. Shao, X., et al. (2025). "Domain adaptation, self-supervision, and generative augmentation enhance GNNs for breast cancer prediction." *Scientific Reports* 15:32924. → **Pathway transfer better than genes (cross-cohort)**

3. Tian, X., et al. (2023). "Cross-platform normalization enables machine learning model training on microarray and RNA-seq data simultaneously." *Communications Biology* 6:588. → **Batch correction enables cross-platform training**

4. Hänzelmann, S., et al. (2013). "GSVA: gene set variation analysis for microarray and RNA-Seq data." *BMC Bioinformatics* 14:7. → **Pathway scoring platform-agnostic**

5. Desmedt, C., et al. (2021). "Single cell RNA-seq data and bulk gene profiles reveal a novel signature of disease progression in multiple myeloma." *Cancer Cell International* 21:190. → **Bulk purity confounding; single-cell insights**

6. Kuiper, R., et al. (2012). "High-Risk Newly Diagnosed Multiple Myeloma: Identification and Prognostication." *Blood* 119(12):2730-2735. → **EMC92/SKY92 develops risk signature**

7. Shaughnessy Jr, J.D., et al. (2007). "A validated gene expression model of high-risk multiple myeloma is defined by deregulated expression of genes mapping to chromosome 1." *Blood* 109(6):2276-2284. → **GEP70 foundational signature**

8. IMWG/IMS Consensus (2025). "International Myeloma Society/International Myeloma Working Group Consensus Recommendations on the Definition of High-Risk Multiple Myeloma." *Journal of Clinical Oncology*. → **Current clinical gold-standard; acknowledges transcriptomic integration**

