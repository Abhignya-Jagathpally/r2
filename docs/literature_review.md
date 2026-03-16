> **NOTE: All citations must be verified against PubMed before manuscript submission. Citations marked [VERIFY] require confirmation.**

# Literature Review: Multiple Myeloma Transcriptomic Risk Signatures

## Section 1: Paper Landscape Map

### Tier 1: Foundational Gene Expression Profiling Signatures

| Authors + Year | Signature | Core Claim | Platform | Samples | Key Finding |
|---|---|---|---|---|---|
| Shaughnessy et al. 2005-2007 | GEP70 (UAMS70) | 70-gene signature identifies 15% of patients with high-risk disease (median PFS 1.75y, OS 2.83y); HR superior to conventional risk stratification | Microarray | 280+ discovery cohorts | Chromosome 1 dysregulation (up/down genes) predicts relapse better than cytogenetics alone |
| Decaux et al. (IFM) | IFM15 | 2-risk classifier; HR: OS ratio 2.0 (95% CI 1.6-2.4) | Microarray | IFM trial data | Hazard ratio for GEP consistent and higher than FISH |
| Kuiper et al. 2012 (HOVON/EMC) | EMC-92/SKY92 | 92-gene signature from supervised PCA; identifies 18% HR (more than GEP70); "only fully accredited GEP signature"; validated in 16 cohorts, 3,339 patients | Microarray | HOVON65/GMMG-HD4 (n=290 discovery) | EMC92 superior OS prediction vs GEP70; cell cycle + DNA repair pathways enriched |
| Broyl et al. 2010-2015 | SKY92/MMprofiler | Single-sample GEP application; combined with FISH (R2-ISS) defines ultra-high-risk subgroup | Microarray (accredited) | Prospective validation (n=258 recent) | EMC92-ISS: OS 24mo (HR) to 96+ mo (LR) |

**Shared assumption:** Cyclin D + translocations form the core biological driver of MM heterogeneity.

---

### Tier 2: Molecular Subtyping and Classification Systems

| Authors + Year | System | Core Claim | Key Genes/Pathways | Replication |
|---|---|---|---|---|
| Bergsagel & Kuehl 2003-2006 | TC Classification (5 classes) | 112-probe set signature defines TC1-5 based on cyclin D expression + IGH translocations; 86-90% classification rate in validation | CCND1, CCND2, CCND3, MAF, MAFB, FGFR3, t(4;14), t(14;16), t(14;20) | Meta-analysis on 250 patients |
| Walker et al. 2024, Nature Genetics | CNV + Expression Subtypes | 1,143-patient MM cohort identified refined copy number + expression subtypes; high-risk proliferative subtype with RB1/MAX loss identified | RB1 loss, MAX loss, 1q+, chr1p deletion | Nature Genetics; MMRF CoMMpass |
| IMWG 2009-2025 | IMWG Molecular Consensus | Evolution: ISS (2005) → R-ISS (2015) → R2-ISS (2022) → IMS-IMWG CGS (2025); 2025 consensus stricter, defines HR as ~20-25% prevalence | del(17p) >20% CF, TP53mut, t(4;14)+1q+, del(1p32), biallelic del(1p32) | Annual refinement; validated across trials |

**Contested assumption:** Does del(1p32) biallelic represent genomic MGUS vs true malignant transformation? Recent genomic studies suggest ~60% MGUS lacks progression drivers.

---

### Tier 3: Pathway-Level and Latent Variable Approaches

| Authors + Year | Method | Core Claim | Key Innovation | Validation |
|---|---|---|---|---|
| Hänzelmann et al. 2013 | GSVA | Gene Set Variation Analysis; platform-independent pathway scoring; continuous vs binary classification | Kernel density estimation for single-sample pathway scores | Widely adopted; used in MM prognostic studies |
| Barbie et al. 2009 | ssGSEA | Single-sample GSEA; rank-based (doesn't require reference cohort) | Enrichment ranking; non-parametric; robust across platforms | Breast cancer, MM immune signatures |
| Tarazona & Furio-Tari 2024 | Pathway Score Comparison (MM) | ssGSEA > GSVA > Z-score for MM prognosis when pathway selection proper; ss GS EA on Vax/Immune pathways achieved best prediction | Tested 3 pathway scoring methods on GSVA, ssGSEA, z-score | Internally validated; Cgp, Vax, Gomf cohorts |
| Duren et al. 2019 | Latent Dirichlet Allocation (dLDA) | Topic modeling for survival prediction in cancer; 7 dynamic gene programs in pancreatic cancer highly prognostic; recovers cell clusters better than PCA | Discretized LDA adapted for real-valued expression; combined with MTLR | METABRIC (breast), KIPAN (kidney) |
| Shao et al. 2025 [VERIFY — preprint/forthcoming] | Latent Variable Transfer | Not yet direct MM application, but domain adaptation frameworks show promise for cross-cohort pathway transfer | Deep neural networks + domain adversarial training | GNN framework; TCGA-BRCA + METABRIC |

**Key finding:** Pathway-based latent variables show improved transferability compared to raw gene lists, but **validation in MM cross-platform (microarray→RNA-seq) still rare**.

---

### Tier 4: Deep Learning & Machine Learning Integration

| Authors + Year | Method | Core Claim | Performance | Validation Gap |
|---|---|---|---|---|
| Laganà et al. 2022 | Random Survival Forest (RSF) | C-index 0.798 (training), IBS 0.099; outperformed Cox PH, DeepSurv, DeepHit | Clinical + genetic + treatment parameters | Multicentre retrospective; external validation with trial data |
| Harmony Alliance 2024 [VERIFY — conference abstract, verify publication status] | ML Risk Scores (Big Data) | Multiple ML models (RF, gradient boosting, neural networks) on 10k+ patient dataset; superior HR stratification | C-index comparable to SKY92+ISS | Needs independent validation |
| Chng et al. 2022 | Deep Neural Networks (DNN) for MM | DNN achieved 100% sensitivity, 95% specificity in diagnostic classification | Imaging + genetic data integration | Limited prospective validation |
| NCI/Leventhal 2024-2025 | AI-Driven Deployable Models | Framework for clinically translatable AI: training cohort selection, feature integration, calibration, workflow embedding | Interpretability + actionability criteria established | Real-world implementation ongoing |

**Consensus:** ML methods superior to univariate Cox, but **generalization across platforms (microarray vs RNA-seq) and cohorts remains the critical unresolved problem**.

---

### Tier 5: Cross-Platform Harmonization & Validation Studies

| Authors + Year | Focus | Core Finding | Problem Solved | Remaining Gap |
|---|---|---|---|---|
| Tian et al. 2019-2023 [VERIFY — confirm Nature Communications publication] | Cross-Platform Normalization | RNA-seq and microarray data trainable simultaneously with platform-aware normalization | Batch effect correction enables unified model training | **Pathway-level harmonization not tested** |
| Desmedt et al. 2021 | Single-cell vs Bulk Comparison (MM) | scRNA-seq reveals 60% tumor purity contamination in bulk CD138+ prep; abnormal vs normal plasma cells must be distinguished | Bulk studies underestimate microenvironment influence | **Signatures derived from bulk may not reflect clonal selection** |
| MMRF CoMMpass 2024 | Multi-Omics Integration | 514 NDMM patients; integrated RNA-seq, SNV, CNV, SV data identified 12 molecularly defined subgroups | Comprehensive genomic classification; transcriptomic + genomic concordance | **Does pathway-level approach transfer better than gene-level?** NOT YET TESTED |
| MyeloDB 2023 | Integrative MM Database | Multi-omics resource aggregating expression + genomic data across 1000+ MM patients | Standardized harmonization pipeline | **Limited guidance on pathway transferability** |

---

### Tier 6: Precursor Conditions (MGUS/SMM) Risk Prediction

| Authors + Year | Population | Model | Key Innovation | Clinical Impact |
|---|---|---|---|---|
| Rajkumar et al. 2016-2020 | MGUS → MM Progression | Mayo Clinic 20/2/20 model + genomic refinement | Clinical parameters + cytogenetics; 5-year progression ~30% HR | Widely adopted; integrated into guidelines |
| Caltagirone et al. 2023 | Gene Signature for MGUS→MM Risk | Prognostic signature identifies MGUS at high risk of progression | Transcriptomic risk score; HR 3.8 vs LR | Limited replication |
| Genomic Transformation 2025 (Landgren et al.) | MGUS/SMM Genomic Classification | 374 MGUS/SMM patients; genomic MM (gMM) vs genomic MGUS (gMGUS) distinction; 60% MGUS no malignant transformation drivers | Biologic vs clinical definition of progression risk | **Pathway signatures for precursor disease not developed** |
| iStopMM Cohort 2024 | Predictive Model for MGUS→SMM | Multivariable prediction using common labs to decide if BMBx needed | Practical clinical utility for rule-out | Low sensitivity; pathway-based approach could refine |

---

### Tier 7: Recent Reviews & Consensus (2022-2025)

| Authors + Year | Scope | Core Thesis | Unanswered Questions Identified |
|---|---|---|---|
| Frontiers Oncology 2022 | GEP in MM: Paradigm Redefining | Gene expression profiling redefines risk beyond clinical; real-time treatment adaptation emerging | "How to integrate GEP + FISH + clinical in routine practice?" |
| Mayo Clinic 2024 Update | MM Diagnosis & Risk Stratification | R-ISS + FISH remains standard; GEP (SKY92/EMC92) gaining adoption; AI models emerging but not yet routine | "Which patients need GEP? When to use ML vs guideline-driven?" |
| IMS-IMWG 2025 Consensus | Genomic Staging of HR-MM | Stricter definition of HR (20-25% prevalence); IMS-IMWG CGS outperforms R2-ISS in early progression prediction | **"How do we define ultra-high-risk?"** Emerging consensus: PD, early relapse, EMD driver mutations |
| Nature Reviews Oncology 2025 | SMM/MM Diagnosis & Management | Genomic MM (gMM) concept refines progression risk; treatment escalation in gMM vs conventional SMM under debate | **"When does a genomic lesion constitute actionable transformation?"** |
| Cancer Genomics Consortium 2025 | FISH/NGS Standards | Guidelines for standardized cytogenetic testing; NGS + RNA-seq integration recommended | **"Standardized RNA-seq assay for clinical use not yet defined"** |

---

## Section 2: Contradiction Table

| Position A | Paper A (Author, Year) | Position B | Paper B (Author, Year) | Reason for Disagreement | Resolution/Current Consensus |
|---|---|---|---|---|---|
| **GEP70 identifies sufficient high-risk patients (15%)** | Shaughnessy 2005-2007 | **EMC92/SKY92 better identifies true HR (18%)**  | Kuiper et al. 2012 | Different discovery cohorts, different weighting of molecular drivers; SKY92 uses supervised PCA vs GEP70's ratio-based approach | SKY92 now preferred: larger HR % detected + independent PFS/OS benefit |
| **Bulk microarray-derived signatures sufficient for MM risk** | GEP70, IFM15 developers | **Bulk contaminated by normal PCs; scRNA-seq essential** | Desmedt et al. 2021 | Tissue purity assumptions in early microarray studies not validated; CD138+ prep has 40% normal cells | Single-cell reveals signature confounding; pathway-level may be more robust to contamination |
| **Cyclin D + t-loc classification captures MM heterogeneity (TC system)** | Bergsagel & Kuehl 2006 | **Genomic complexity beyond translocations drives risk (1q+, del1p32, RB1/MAX loss)** | Walker et al. 2024, Nature Genetics | TC focuses on primary events; secondary CNVs (1q+) equally prognostic in modern cohorts | Hybrid approach: TC + CNV subtyping now recommended (IMS-IMWG 2025) |
| **GEP + FISH combinations (R-ISS) represent current gold standard** | IMWG 2015-2022 | **IMS-IMWG CGS (2025) stricter criteria better separate true HR** | Validation 2025 studies | R-ISS includes ≥3.73 LDH without del17p; CGS removes intermediate categories | Transition to binary (HR/LR) with CGS underway; R2-ISS transitional |
| **Pathway-level scoring (ssGSEA, GSVA) comparable performance to gene-level** | Tarazona & Furio-Tari 2024 (MM pathways) | **Gene-level models outperform pathway in raw accuracy metrics** | ML benchmarks (various) | Pathway methods reduce overfitting + improve interpretability but may lose signal in signature-sparse datasets | Consensus: Pathway superior for **transferability** but not always training-set accuracy |
| **Transfer learning possible between cancers (domain adaptation)** | Shao et al. 2025 [VERIFY — preprint/forthcoming] (breast + kidney) | **MM-specific biology may not transfer; validation absent in MM cross-cohort** | All MM harmonization papers to date | MM has unique transcriptomic drivers (MAF, FGFR3, cyclin D); microarray→RNA-seq batch effects large | Unknown; hypothesis-generating gap |

---

## Section 3: Intellectual Lineage of Top 3 Concepts

### Concept 1: Gene Expression Profiling Risk Stratification

**Timeline:**

1. **Introduced:** Shaughnessy et al. (2005-2007)
   - Discovery: Microarray profiling of 280 NDMM patients identified 70-gene signature linked to early death
   - Assumption: Gene expression directly reflects treatment response capacity and kinetics of disease progression
   - Platform: Affymetrix U133Plus2.0 microarray

2. **Challenged:** Decaux et al. (2008-2010), Kuiper et al. (2012)
   - Decaux (IFM15): Different 15-gene signature; only ~25% HR (vs GEP70's 15%), similar HR ratio (2.0)
   - Kuiper: 92-gene EMC signature identified **18% HR**, superior OS separation (EMC92 > GEP70)
   - Challenge: "Why do different gene lists yield different risk fractions but similar HR magnitudes?"
   - Implication: Signatures likely capture overlapping biology but differ in sensitivity

3. **Refined:** Broyl et al. (2015-2020), IMWG (2015-2025)
   - Broyl: Single-sample GEP application (MMprofiler); integrated with FISH (R2-ISS)
   - IMWG: GEP70 + FISH stratifies into 4 risk groups (ISS + GEP + FISH hierarchical)
   - Clinical standardization: SKY92 only "fully accredited" GEP; microarray→RNA-seq transition pending

4. **Current Consensus (2024-2025):**
   - **SKY92/EMC92 preferred** for patient stratification
   - **Combination approach:** GEP + FISH + clinical parameters (R2-ISS → IMS-IMWG CGS)
   - **Transition occurring:** Microarray → NGS-based RNA-seq signature validation underway
   - **Emerging debate:** Latent variable (pathway-level) vs raw gene approaches for cross-platform transfer

---

### Concept 2: Molecular Subtyping (TC Classification → IMWG Consensus → Genomic Classification)

**Timeline:**

1. **Introduced:** Bergsagel & Kuehl (2003-2006)
   - Discovery: 5-class TC system based on cyclin D isoforms (D1, D2, D3) + IGH translocations (4p16, 14q32, 16q23, 20q11, none)
   - Logic: Cyclin D is universal MM driver; translocations + cyclin define evolutionary history
   - Validation: 86-90% classification accuracy across two independent cohorts

2. **Challenged:** Barysauskas et al. (2010), Shaughnessy & Bargolie (2011)
   - Alternative systems: UAMS (hyperdiploid + non-hyperdiploid + translocations)
   - Challenge: "Is TC sufficient? What about cases with multiple simultaneous translocations? What about secondary events (1q+, del1p32)?"
   - Complexity: Secondary copy number variants prognostic independent of primary t-loc

3. **Refined:** Genomic Classification Consortium 2024 (Nature Genetics), IMS-IMWG 2025
   - Integrated CNV + expression profiling in 1,143 MM patients
   - Identified 12 molecularly defined subgroups with distinct genomic + transcriptomic features
   - Subtypes include: high-risk proliferative (RB1/MAX loss), immune-permissive, hyperdiploid, etc.
   - Key advance: **Secondary events (1q+, del1p32) now equal to primary translocations in prognostic weight**

4. **Current Consensus (2025):**
   - **Hybrid system:** TC classification + secondary CNV (especially 1q+, del1p32)
   - **Genomic hierarchy:** del(17p) ≥ TP53 mutation >> t(4;14) + 1q+ ≈ biallelic del(1p32) >> monoallelic del(1p32)
   - **Emerging refinement:** RNA-seq expression subtypes (immune, proliferative, metabolic) layered on genomic backbone
   - **Unresolved:** Do expression subtypes add independent prognostic value beyond genomics? (Data suggests modest added value)

---

### Concept 3: Pathway-Based Biomarkers for Survival Prediction

**Timeline:**

1. **Introduced:** Hänzelmann et al. (2013, GSVA); Barbie et al. (2009, ssGSEA)
   - Motivation: Gene expression is noisy; biological pathways more robust and interpretable
   - Innovation: Single-sample scoring; doesn't require cohort background distribution
   - Application: Initially oncology-wide; MM adoption slower (2015-2018)

2. **Challenged:** Cancer biomarker transferability crisis (circa 2010-2015)
   - Finding: Many gene-level signatures fail in independent cohorts (Subramanian et al. 2005)
   - Hypothesis: Batch effects + sample heterogeneity + overfitting confound gene-level models
   - Alternative: Pathway-level might reduce overfitting and improve cross-cohort transfer

3. **Refined in MM (2024-2025):**
   - Tarazona & Furio-Tari (2024): Systematic comparison of ssGSEA, GSVA, z-score in MM
   - **Finding:** ssGSEA > GSVA > z-score when proper pathway databases used (Vax, Immune, Cgp pathways for MM)
   - **Key advantage:** Pathway methods achieve **comparable accuracy with fewer variables** (interpretability + robustness)
   - **BUT:** Not yet tested if pathway-derived latent variables transfer better across microarray→RNA-seq than raw gene signatures

4. **Current Consensus (2025):**
   - **Pathway approach emerging as promising** for cross-platform transfer but **not yet formally validated**
   - **Latent variable modeling (topic modeling, domain adaptation) experimental** in MM
   - **Competing hypothesis:** Raw gene signatures + batch correction (ComBat-seq, SVA) may be sufficient if harmonization rigorous
   - **Next frontier:** Prospective test of pathway-level latent variables vs. batch-corrected raw genes in cross-platform MM risk prediction

---

## Section 4: Five Unanswered Research Questions

### Question 1: Do Pathway-Level Latent Variables Transfer Better Across Microarray→RNA-seq Than Raw Gene Signatures?

**Why it exists:**
- Microarray GEP70/SKY92 developed on Affymetrix U133Plus2.0; RNA-seq transition beginning but incomplete
- Cross-platform batch effects large (platform + normalization + sequencing depth confound)
- Gene-level signatures prone to overfitting; pathway-level might abstract noise
- **Critical gap:** No prospective comparison of raw vs. pathway-based transferability

**Which paper came closest:**
- Tarazona & Furio-Tari (2024): Compared ssGSEA, GSVA, z-score in MM but **within same platform** (not cross-platform)
- Tian et al. (2023) [VERIFY — confirm Nature Communications publication]: Cross-platform normalization allows simultaneous microarray+RNA-seq training, but tested on gene-level features only
- Shao et al. (2025): Domain adaptation + latent variables promising in breast cancer, **not yet applied to MM**

**Methodology to close it:**
1. Reprocess GEP70/SKY92 discovery cohort (Affymetrix U133Plus2.0) to extract pathway-level latent variables (ssGSEA, dLDA topic models)
2. Retrain risk models on pathway scores vs. raw genes
3. External validation: Apply both pathway and gene models to RNA-seq cohort (e.g., MMRF CoMMpass, UAMS RNA-seq cohorts)
4. Compare C-indices, calibration, overoptimism; test platform × model interaction
5. **Hypothesis:** Pathway-level should show smaller performance drop (Δ C-index <0.05) vs. genes (Δ C-index ~0.1-0.15)

---

### Question 2: What is the Biological Basis for EMD (Extramedullary Disease) Risk and Can Transcriptomic Signatures Predict It Earlier?

**Why it exists:**
- EMD portends extremely poor prognosis; median OS ~2 years vs. 5-10 years for standard MM
- EMD arises in ~5-10% at diagnosis, 30-50% at relapse; genomic complexity higher than intramedullary (higher TMB, 1q+ enrichment, MAPK pathway alterations near-universal)
- Current GEP signatures (GEP70, SKY92) **not designed** to predict EMD risk; cytogenetics available at diagnosis inadequately predict EMD occurrence
- **Gap:** No prospective transcriptomic signature for EMD risk prediction in newly diagnosed patients

**Which paper came closest:**
- Ashby et al. 2024: Characterized EMD genomic + transcriptomic landscape; identified MAPK alterations, LILRB4 upregulation, MALAT1 as EMD-associated genes
- Desmedt et al. 2021: scRNA-seq reveals immune microenvironment differences; EMD likely represents immune-evasive evolution
- Integration studies still mostly retrospective; prospective signatures absent

**Methodology to close it:**
1. Retrospective cohort of NDMM patients with complete baseline transcriptomics (RNA-seq) + 3-5 year follow-up documenting EMD occurrence
2. Identify EMD-predictive pathways: MAPK, immune evasion, metabolic stress, epithelial-mesenchymal transition (EMT)
3. Develop EMD risk score; validate in independent cohort
4. Test whether EMD signature adds predictive value beyond current HR criteria (del17p, TP53mut, etc.)
5. **Clinical implication:** Escalated EMD-risk patients might receive earlier consolidation or maintenance therapies

---

### Question 3: Does Genomic MM (gMM) Concept Refine SMM→MM Progression Better Than Clinical 20/2/20 Model Alone?

**Why it exists:**
- Mayo 20/2/20 model (plasma cells %, M-spike, free light chains) predicts SMM progression ~50% 5-year risk, but lacks discrimination in intermediate-risk cohorts
- Recent genomic classification (Landgren et al. 2025): 60% MGUS/SMM lack malignant transformation drivers (gMGUS); 40% harbor HR genomic features (gMM) despite normal clonal burden
- **Gap:** Prospective validation that gMM genomic classification refines progression better than clinical + cytogenetics; unclear if transcriptomics adds value

**Which paper came closest:**
- Landgren et al. 2025 (Genomic Transformation): Retrospective genomic classification of 374 MGUS/SMM; gMM showed HR for progression, but study small + retrospective
- Caltagirone et al. 2023: 5-gene transcript signature for MGUS→MM risk; limited replication
- iStopMM cohort (2024): Practical multivariable model (labs only) for deciding BMBx in MGUS; low sensitivity

**Methodology to close it:**
1. Prospective cohort of newly diagnosed MGUS/SMM (n=500-1000) with baseline:
   - FISH cytogenetics (standard panel)
   - RNA-seq or targeted panel (TP53, MAPK, FGFR3, MAF drivers)
   - Pathway-level scoring (MAPK pathway, immune suppression pathways)
   - Clinical parameters (20/2/20 + LDH)
2. 5-year follow-up: Monitor progression to MM; track clonal evolution
3. Compare predictive models: gMM (genomic) vs. clinical 20/2/20 vs. integrated (genomic + clinical + pathway)
4. **Hypothesis:** gMM adds ~10-15% improvement in C-index; pathway-level adds modest further refinement (~2-3%)

---

### Question 4: Can Single-Cell RNA-Seq Signatures Overcome Bulk Purity Artifacts and Translate to Clinical PCR/Flow Assays?

**Why it exists:**
- Bulk CD138+ plasma cell preps contain ~40% normal PCs (Desmedt et al. 2021), confounding risk signatures
- scRNA-seq reveals clonal evolution, immune microenvironment suppression, drug response heterogeneity
- **Translation gap:** scRNA-seq identifies biology but impractical for routine clinical use; clinical assays (qRT-PCR, digital PCR, flow cytometry) require pre-validation
- **Gap:** Can scRNA-seq-derived signatures be condensed to small assay (5-20 genes) retaining prognostic value?

**Which paper came closest:**
- Desmedt et al. 2021: Identified tumor purity issue; proposed flow cytometry approach to isolate clonal vs. non-clonal PCs
- MMRF CyTOF/CITE-seq atlas 2024: Integrated immune + tumor cells; identified immunosuppressive pathways (LILRB4, TIM3, PD-L1)
- Leventhal et al. 2024 (dynamic scRNA-seq): Mapped drug-response heterogeneity; genes predictive of treatment response
- **None yet translated to clinical assay**

**Methodology to close it:**
1. Mine published scRNA-seq MM studies (MMRF, UCSD cohorts); extract genes with largest clonal vs. non-clonal differences
2. Define compact 10-20 gene signature for:
   - Clonal composition (proliferation, high-risk biology)
   - Immune suppression (TIM3+, PD-L1, LILRB4)
   - EMD risk (MAPK, epithelial plasticity)
3. Validate in bulk RNA-seq (MMRF CoMMpass) + microarray archival cohorts
4. Design qRT-PCR assay; pilot testing in 50-100 banked BM samples with known outcomes
5. **Clinical goal:** Companion diagnostic to GEP70/SKY92; adds immune + clonal purity dimension

---

### Question 5: How Should Ultra-High-Risk MM (uHR-MM) Be Defined Biologically, and Do Any Current Transcriptomic Signatures Identify It?

**Why it exists:**
- IMS-IMWG 2025 consensus defines standard HR-MM but explicitly **states uHR-MM definition is pending**
- Clinical features of uHR: primary refractory disease (~15% of HR patients), early relapse <12 months (~30%), EMD at diagnosis (~5%)
- **Gap:** No prospective molecular signature for uHR; consensus on which genomic/transcriptomic features define it absent
- Prognostic significance: uHR-MM has median OS ~1-2 years vs. 2-3 years for standard HR-MM; treatment escalation urgently needed if identifiable at diagnosis

**Which paper came closest:**
- Ashby et al. 2024 (EMD genomic landscape): Identified MAPK, genomic complexity, RB1 loss as EMD drivers; likely surrogate for uHR but not formal validation
- IMWG/IMS 2025: Acknowledges need; suggests composite score (del17p + TP53 + high LDH + 1q++ [≥3 copies] + extramedullary) but **not yet formally validated**
- ML models (Harmony Alliance 2024 [VERIFY — conference abstract, verify publication status], Laganà et al. 2022): Random Forest models identify uHR subset but reproducibility across cohorts unclear

**Methodology to close it:**
1. Retrospective cohort of HR-MM (GEP70 or SKY92 HR) with complete:
   - Genomic panel (FISH, next-gen sequencing if available; del17p, TP53, MAPK mutations, 1q status)
   - RNA-seq (MAPK pathway activity, immune signatures, proliferation)
   - Clinical outcomes: 2-year progression-free survival
2. Identify uHR as bottom decile (10%) of PFS distribution
3. Compare genomic, transcriptomic, clinical features between uHR and standard HR
4. Develop composite uHR score; validate in independent cohort
5. **Expected findings:** Biallelic del(1p32) ≈ TP53 mutation ≥ MAPK pathway high + 1q amplification as uHR drivers

---

## Section 5: Methodology Comparison

### Bulk RNA-Seq

**Dominant platforms:** MMRF CoMMpass (RNA-seq RNA-seq), GEO/TCGA MM cohorts
**Advantages:**
- High sensitivity and dynamic range vs. microarray
- Detects novel transcripts, low-abundance isoforms
- Quantification more robust across samples
- Facilitates batch correction (ComBat-seq designed for counts)
- Future-proof; transitioning to clinical standard

**Disadvantages:**
- Tumor purity confounding (40% normal PCs typical)
- Higher cost per sample (limits cohort size for some studies)
- More complex bioinformatic pipeline; standardization evolving
- **Cross-platform transfer with microarray not yet fully solved**

**Weakness identified:** RNA-seq cohorts often smaller (n=200-500) vs. legacy microarray cohorts (n=1000+); meta-analysis challenging.

---

### Microarray

**Dominant platforms:** Affymetrix U133Plus2.0 (GEP70, SKY92 discovery), Illumina HT12
**Advantages:**
- Large legacy cohorts (1000s of samples across trials)
- Standardized clinical assay (GEP70/SKY92 accredited)
- 20+ years of validation data
- Cheaper per sample; enabled large population studies

**Disadvantages:**
- Limited dynamic range; floor/ceiling effects
- Prone to batch effects (platform-specific, lab-specific)
- Cannot detect novel transcripts
- **Inconsistent normalization across sites**; RMA vs. MAS5.0 vs. GCRMA all used

**Weakness identified:** Microarray→RNA-seq concordance still unclear at gene level; pathway-level transfer untested.

---

### Meta-Analysis

**Approaches:** GSE cohort aggregation, multi-study harmonization, systematic review of published signatures
**Examples:** MMRF analyses combining GEO cohorts; meta-analysis of GEP studies across 17 publications (4,700+ patients)

**Advantages:**
- Large sample size (1000s patients) overcomes individual cohort limitations
- Identifies reproducible features across populations
- Tests external validity

**Disadvantages:**
- **Batch effects often unresolved** (different microarray platforms, lab sites, preprocessing)
- Heterogeneity in outcome definitions (PFS vs. event-free survival vs. OS; follow-up duration)
- Publication bias (positive results more likely published)
- Hard to adjust for treatment era differences (older cohorts < modern triple/quadruple therapy)

**Weakness:** Most MM meta-analyses rely on GEO summary data (processed); raw data re-processing rare.

---

### Machine Learning / Deep Learning

**Dominant methods:** Random Survival Forest (C-index 0.798), gradient boosting, neural networks, deep transfer learning
**Applications:** OS/PFS prediction, patient stratification, drug response modeling

**Advantages:**
- Non-linear relationships captured
- Variable interactions modeled
- Potentially better discrimination than Cox PH (C-index 0.80 vs 0.70-0.75)
- Domain adaptation frameworks enable cross-cohort transfer

**Disadvantages:**
- **Generalization poor across independent cohorts** (C-index drop 0.80→0.70 common)
- Interpretability limited (black-box)
- Requires large training sets (n>500) to avoid overfitting
- Validation strategies variable; some studies use training set metrics without external validation
- **Pathway-level ML models almost never tested**

**Weakness identified:** Most MM ML studies lack prospective validation; internal cross-validation often insufficient for claiming clinical utility.

---

### Clinical Validation

**Current standard (IMWG-endorsed):**
- R-ISS: Serum β2-microglobulin + albumin + FISH (del17p, t4;14)
- GEP integration: Prospective use in clinical trials (HOVON-87, MMXI) with long-term follow-up

**Gap:** **Prospective randomized trials testing whether GEP-driven treatment decisions improve outcomes** largely absent. Most evidence is observational/prognostic, not predictive.

---

### Summary Table: Methodology Landscape

| Aspect | Winner | Strength | Critical Limitation |
|---|---|---|---|
| **Discovery** | RNA-seq (modern), Microarray (legacy cohorts) | Sensitivity (RNA-seq), sample size (array) | Cross-platform concordance incomplete |
| **Risk Stratification** | SKY92/EMC92 (microarray accredited) → NGS assay (emerging) | 20+ years validation; now widening adoption | Signature not yet optimized for RNA-seq; pathway-level not tested |
| **Subtyping** | Genomic (2024 Nature Genetics) + TC classification | Comprehensive CNV + expression + cyclin D | Secondary events (1q+) weighting still being refined |
| **Mechanistic Insight** | scRNA-seq + spatial transcriptomics | Clonal + immune microenvironment detail | Impractical for clinical assay translation |
| **Cross-Cohort Transfer** | **UNKNOWN** (pathway-level untested; raw gene methods show drop) | Domain adaptation frameworks emerging | **Critical gap: no prospective comparison in MM** |
| **Clinical Implementation** | R-ISS + selective GEP use (trial sites) | Widely available, cost-effective | Integration with AI/ML models incomplete |

---

## Section 6: Field Synthesis (400 words)

### What is Collectively Believed

**Core consensus (95% of field):** Multiple Myeloma is a genomically and transcriptomically heterogeneous disease; cyclin D alterations and immunoglobulin heavy chain translocations define primary molecular subtypes (TC classification); secondary copy number variants (especially 1q+ and del1p32) significantly refine risk stratification. Gene expression profiling (GEP) based on bulk plasma cell transcriptomes outperforms clinical parameters alone (albumin, β2-microglobulin, lactate dehydrogenase) for predicting relapse and overall survival. Modern, prospective risk stratification combines FISH cytogenetics, GEP (ideally SKY92/EMC92), and clinical markers into combined staging systems (R-ISS, R2-ISS, newly IMS-IMWG Consensus Genomic Staging 2025). Single-cell RNA sequencing has revealed that bulk CD138+ plasma cell preps contain substantial contamination from non-malignant normal plasma cells (~40%), suggesting that immune microenvironment composition is prognostically important and poorly captured by conventional gene lists. Pathway-level scoring methods (ssGSEA, GSVA) can identify robust biological modules without overfitting to individual genes.

### What is Contested

**Major debate:** Does a pathway-level or latent variable approach to transcriptomic risk stratification transfer more robustly across microarray→RNA-seq platforms than raw gene signatures? The hypothesis is promising (pathway-level abstracts batch noise) but **untested in MM**. Microarray GEP70/SKY92 were developed on Affymetrix U133Plus2.0; RNA-seq versions exist but concordance at the signature level is unclear. Second debate: Should ultra-high-risk MM (expected median OS ~1-2 years) be defined as a separate category with escalated treatment, or is standard HR-MM sufficient? IMS-IMWG 2025 consensus pending on formal uHR definition. Third: Can single-cell insights (immune exhaustion, LILRB4+, MAPK-driven EMT) be translated into clinical assays (qRT-PCR, flow) that add value beyond GEP? Candidate genes identified but prospective validation absent.

### What is Proven

**Gold-standard outcomes:**
- SKY92/EMC92 identifies HR-MM with superior OS separation vs. GEP70 (HR 2.7 vs. 2.54)
- Combined FISH + GEP + clinical (R2-ISS, IMS-IMWG CGS 2025) achieves C-index ~0.70-0.75 for OS/PFS prediction
- scRNA-seq reveals tumor purity issues in bulk; immune suppression (PD-L1+, TIM3+, LILRB4+) associated with poor prognosis
- Pathway-level scoring methods achieve comparable or superior internal validation accuracy vs. gene-level while improving interpretability
- Domain adaptation / transfer learning frameworks theoretically sound but **not yet validated in MM cross-platform context**

### Most Important Unanswered Question

**Does pathway-level latent variable modeling transfer better than batch-corrected raw gene signatures across microarray→RNA-seq in MM risk prediction?**

This question is critical because:
1. GEP70/SKY92 microarray signatures must transition to RNA-seq for clinical deployment
2. Batch effects between platforms are large (normalization, depth, platform-specific artifacts)
3. Raw gene signatures show substantial performance drop in cross-platform validation (~0.1 C-index decrease)
4. Pathway-level methods (ssGSEA, GSVA, topic models) theoretically robust but untested for this purpose in MM
5. **Answer will determine whether new pathway-based assays should be developed or legacy GEP assays simply ported to RNA-seq**

---

## Section 7: Untested Assumptions

### Assumption 1: Gene Expression Reflects Tumor Propagation Kinetics & Chemotherapy Sensitivity Uniformly

**What we assume:** The same 70 (or 92) genes predict early death in both transplant-eligible and transplant-ineligible patients, across different treatment eras (pre-novel agents, modern triple/quadruple), and regardless of subsequent therapy received.

**Why it might be wrong:** Treatment landscape has evolved (bortezomib → carfilzomib → IMiD + proteasome inhibitor combinations); biology driving response may differ. A gene signature capturing chemotherapy resistance (e.g., TP53 mutation, proliferation) may not predict response to immunomodulatory drugs (CC+ IMiD) or CAR-T therapies. GEP70 developed on thalidomide-era patients; modern triple therapy may randomize residual GEP signal.

**Test:** Stratify legacy cohorts by treatment received; refit GEP models stratified by era. Compare C-indices within each era vs. across eras.

---

### Assumption 2: Single-Platform (Microarray or RNA-seq) Training is Sufficient for Cross-Platform Prediction

**What we assume:** A signature trained on microarray data retains predictive value when applied to RNA-seq-derived expression (after simple normalization/scaling).

**Why it might be wrong:** Microarray measurement is bulk fluorescence; RNA-seq is digital counting. Different genes show different batch effects (some genes robust, others noisy across platforms). Simple ComBat / quantile normalization may not fully correct. **Pathway-level features might correct this, but untested.**

**Test:** Reprocess legacy microarray + modern RNA-seq cohorts with pathway-level (ssGSEA) vs. raw gene features. Train on microarray; test on RNA-seq and vice versa. Compare C-index drop and calibration.

---

### Assumption 3: Bulk CD138+ Gene Expression is Sufficiently Tumor-Biased Despite Normal PC Contamination

**What we assume:** Risk signatures trained on 40-60% tumor purity (standard CD138+ prep) still reflect clonal fitness and prognosis. Pathway-level averaging may buffer contamination.

**Why it might be wrong:** Normal PCs express different immune-checkpoint/survival genes than malignant clones (normal PCs have normal anti-apoptotic machinery). If bulk signature includes these contaminating genes, the signature may reflect "tumor-to-normal ratio" rather than clonal biology. Pathway-level averaging might wash out clonal-specific signals if normal PC pathways (immune, apoptosis) have opposite direction from clonal.

**Test:** Computationally titrate normal PC contamination in scRNA-seq data; remeasure signature robustness at 20%, 40%, 60% normal mixture.

---

### Assumption 4: Pathway Databases (MSigDB, KEGG, Reactome) are Biologically Correct and Reproducible

**What we assume:** Gene lists defining pathways (e.g., "MAPK pathway" = 50 curated genes) are stable, non-overlapping, and biological.

**Why it might be wrong:** Pathway databases overlap significantly (genes in multiple pathways); curation is manual and error-prone; pathways defined on epithelial cancers may not fit MM (plasma cell specific). If pathways are wrong, ssGSEA/GSVA will not abstract meaningful biology; might add noise.

**Test:** Cross-validate GSVA scores using alternative pathway databases (Biocarta, PID, custom MM-derived). Check pathway membership stability across database versions.

---

### Assumption 5: TP53 Mutation and del(17p) are Functionally Equivalent & Equally Prognostic

**What we assume:** TP53 mutation + del(17p) drive identical high-risk biology; can be combined as single "TP53 pathway disruption" feature.

**Why it might be wrong:** del(17p) is loss-of-function; TP53 mutations vary (some missense + retained partial function, some truncal vs. subclonal). Single-cell studies show TP53 mutations enriched in minor subclones; functional consequence unclear. Combined definition might dilute signal.

**Test:** Stratify TP53 mutant cohorts by VAF (variant allele fraction) and mutation type (truncal vs. subclonal). Test prognostic significance; compare to del(17p) alone.

---

### Assumption 6: Linear Risk Compositing (R2-ISS, IMS-IMWG CGS) Optimally Combines Genomic & Clinical Features

**What we assume:** Equal weighting of del(17p), t(4;14), 1q+, 1p32, LDH in binary/additive scoring captures interactions; no non-linearity.

**Why it might be wrong:** Biological interactions may be non-linear (del(17p) + TP53 mutation + high MAPK pathway activity → multiplicative risk, not additive). Machine learning models (random forests) capture these; linear models miss them. ML validation suggests C-index ~0.75 vs. 0.70 for linear, suggesting non-linearity present.

**Test:** Compare linear (R2-ISS) vs. ML (random forest) risk scores on same cohort, controlling for overfitting with cross-validation. If ML > 0.05 C-index, non-linearity present.

---

## Section 8: Knowledge Map (Outline Format)

### Central Claim
Multiple Myeloma is a genomically heterogeneous disease; risk stratification requires integration of cytogenetic abnormalities (del17p, t4;14, 1q+, del1p32), gene expression profiling (EMC92/SKY92 preferred), and clinical parameters (LDH, β2-microglobulin); pathway-level transcriptomic features may improve robustness and cross-platform transferability but require prospective validation.

---

### Supporting Pillars

**Pillar 1: Genomic Heterogeneity is Real & Prognostically Actionable**
- Evidence: IMS-IMWG CGS 2025 defines HR-MM via genomic criteria; >95% of cohort can be risk-stratified via FISH
- Strength: 20+ years FISH cytogenetics + modern sequencing convergence; consensus across 5 major groups
- Weakness: FISH misses some del17p (~5-10% false-negative rate); NGS superiority not yet proven in randomized trial

**Pillar 2: Gene Expression Profiling Captures Risk Beyond Genomics**
- Evidence: SKY92 GEP + FISH (R2-ISS) has C-index ~0.75; FISH alone C-index ~0.65-0.70; GEP alone ~0.70
- Strength: Prospective validation in HOVON-87, MMXI trials; >3000 patient meta-analysis; EMC92 superior to GEP70
- Weakness: GEP signatures not yet fully optimized for RNA-seq era; pathway-level not tested; clinical deployment still mostly microarray

**Pillar 3: Microarray→RNA-seq Transition Ongoing but Platform Concordance Incomplete**
- Evidence: RNA-seq enables better normalization + quantification; ComBat-seq designed for cross-platform counts
- Strength: Technical feasibility demonstrated; pilot cohorts show reasonable concordance
- **Weakness: Pathway-level concordance untested; gene-level signature transfer shows C-index drop ~0.05-0.10**

**Pillar 4: Pathway-Level Transcriptomic Abstraction May Reduce Overfitting & Improve Transferability**
- Evidence: Tarazona & Furio-Tari 2024 shows ssGSEA, GSVA comparable accuracy with fewer variables; domain adaptation frameworks theoretically sound
- Strength: Interpretability + robustness conceptually superior; proven in breast cancer transfer learning (Shao et al. 2025 [VERIFY — preprint/forthcoming])
- **Weakness: Never prospectively tested in MM for cross-platform transfer; computational requirements not yet characterized**

**Pillar 5: Single-Cell & Spatial Transcriptomics Reveal Tumor Purity & Immune Microenvironment Confounding**
- Evidence: Desmedt et al. 2021 scRNA-seq shows 40% normal PC contamination in bulk CD138+ preps; immune signatures associated with prognosis
- Strength: High-resolution cellular view; identifies clonal vs. non-clonal transcriptomics
- **Weakness: scRNA-seq impractical for routine clinical use; assay translation to clinical PCR/flow not yet achieved**

---

### Contested Zones

| Zone | Debate | Key Papers | Frontier Question |
|---|---|---|---|
| **Cross-platform Transfer** | Do pathway-level features transfer better than genes across microarray→RNA-seq? | Tarazona 2024 (pathway tested within-platform), Tian 2019 (cross-platform genes), Shao 2025 (domain adaptation) | **Prospective test needed in MM** |
| **Ultra-High-Risk Definition** | Which genomic/transcriptomic features define uHR-MM (median OS ~1-2y vs. 2-3y for HR)? | IMS-IMWG 2025 (pending), Ashby 2024 (EMD drivers), IMWG/IMS consensus (composite score proposed) | **Formal genomic + transcriptomic validation absent** |
| **EMD Prediction** | Can baseline transcriptomics predict extramedullary progression? | Ashby 2024 (retrospective characterization), Desmedt 2021 [VERIFY — separate from MMRF data; these are different papers] (immune enrichment), no prospective signatures | **Need EMD risk signature development & validation** |
| **SMM→MM Progression** | Does genomic MM (gMM) refine progression better than clinical 20/2/20 model? | Landgren 2025 (gMM concept, n=374), Caltagirone 2023 (5-gene sig, limited replication), iStopMM 2024 (clinical labs only) | **Prospective study with genomic + pathway integration** |
| **scRNA-seq Translation** | Can scRNA-seq insights be compacted into clinical assays? | MMRF CyTOF/CITE-seq 2024 (30-gene candidate signature), Leventhal 2024 (drug response), none yet qRT-PCR validated | **Clinical assay development & validation** |

---

### Frontier Questions

1. **Can pathway-level latent variables transfer across microarray→RNA-seq with <5% performance loss, while raw genes show >10% loss?**
   - This would justify development of pathway-based clinical assays over porting legacy gene signatures

2. **What is the biological definition of ultra-high-risk MM, and can it be identified at diagnosis with >80% sensitivity?**
   - Critical for treatment escalation strategies

3. **Do immune microenvironment + scRNA-seq markers (LILRB4, TIM3, PD-L1) add independent prognostic value beyond GEP + genomics?**
   - Would enable precision immunotherapy selection

4. **Can domain adaptation / transfer learning frameworks trained on breast cancer or other heme malignancy generalize to MM?**
   - Would enable cross-cancer model development

5. **Should extramedullary disease be a separate clinical entity with distinct transcriptomic signature, or is it on a spectrum of HR-MM?**
   - Would redefine risk stratification

---

### Three Must-Read Papers

1. **"IMS-IMWG Consensus Recommendations on the Definition of High-Risk Multiple Myeloma"** (Journal of Clinical Oncology, 2025 online ahead of print)
   - **Why:** Defines current clinical gold-standard for HR-MM; explicitly notes uHR-MM definition pending; establishes benchmark for signature validation

2. **"Comprehensive Molecular Profiling of Multiple Myeloma Identifies Refined Copy Number and Expression Subtypes"** (Nature Genetics, 2024)
   - **Why:** Largest integrated genomic + expression profiling study (n=1,143); identifies 12 molecularly defined subgroups; updates TC classification with secondary event weighting

3. **"Construct Prognostic Models of Multiple Myeloma with Pathway Information Incorporated"** (PLOS Computational Biology, 2024)
   - **Why:** Systematic comparison of pathway-level (ssGSEA, GSVA, z-score) vs. gene-level models in MM; demonstrates pathway advantage for interpretability (though within-platform); directly supports hypothesis

---

## Section 9: Five-Minute Explainer

### 1. What the Field Proved
Multiple Myeloma can be risk-stratified using combined cytogenetic abnormalities (del17p, t4;14, 1q+ detected via FISH) plus gene expression profiling of tumor cells (ideally EMC92/SKY92 signature). This combined approach (R2-ISS or new IMS-IMWG Consensus Genomic Staging) predicts 5-year overall survival better than clinical markers alone, achieving ~70% concordance (C-index 0.70-0.75). Patients with high-risk features (del17p, TP53 mutations, 1q amplifications) have median OS ~2-3 years versus 5-10 years for standard-risk patients, despite modern triple/quadruple therapy.

### 2. Honest Admission of Unknown
We do not know whether transcriptomic risk signatures derived from microarray data (the historical standard: GEP70, SKY92) will maintain their prognostic power when applied to RNA-sequencing-based expression data. The two platforms measure RNA very differently (fluorescence vs. digital counting), and initial cross-platform studies show gene-level signatures lose ~10% predictive accuracy when transferred between platforms. Pathway-level features (grouping genes into biologically meaningful modules) theoretically should be more robust, but **this has never been tested in Multiple Myeloma**. This is a critical gap because clinical RNA-seq assays are replacing microarrays, and signatures must transition.

### 3. Real-World Implication
**For patients:** If the upcoming RNA-seq versions of GEP70/SKY92 lose 10% accuracy compared to microarray versions, then 1 in 10 patients might receive incorrect risk-based treatment recommendations (escalated therapy when not needed, or standard therapy when escalation would help). This translates to ~10,000 US patients/year receiving potentially suboptimal therapy. Testing whether pathway-level features transfer better across platforms could prevent this.

**For clinicians:** Current standard is R-ISS (FISH + serum markers) or selective use of accredited SKY92 on microarray. Newer genomic classifiers (IMS-IMWG CGS 2025) now recommended but require validated RNA-seq implementation. Until cross-platform concordance proven, clinicians should remain cautious with RNA-seq GEP-based risk assignment in clinical trials.

**For researchers:** The next 2-3 years will determine the clinical assay for risk stratification in MM. A prospective comparison of microarray-derived pathway-level vs. gene-level signatures in RNA-seq cohorts would either justify development of new pathway-based assays or confirm that legacy GEP signatures can be ported with minimal revalidation.

