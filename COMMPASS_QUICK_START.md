# CoMMpass Dataset Integration - Quick Start Guide

## Overview

The enhanced `download_geo.py` now supports the MMRF CoMMpass IA21 RNA-seq dataset alongside the three GEO datasets (GSE19784, GSE39754, GSE2658).

## Quick Start: 3 Steps

### Step 1: Register and Download CoMMpass Files

Visit: https://research.themmrf.org/

1. Create account with institutional email
2. Complete researcher profile
3. Accept Data Use Agreement (DUA)
4. Download these 3 files:
   - `MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv` (gene expression)
   - `MMRF_CoMMpass_IA21_PER_PATIENT.tsv` (clinical metadata)
   - `MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv` (treatment response)

### Step 2: Place Files in a Directory

```bash
mkdir -p ~/data/commpass
# Move downloaded files here
mv ~/Downloads/MMRF_CoMMpass_IA21_*.tsv ~/data/commpass/
```

### Step 3: Run Pipeline with CoMMpass

**Option A: Command Line**
```bash
cd /sessions/sweet-stoic-cray/r2-fresh
python src/preprocessing/download_geo.py --commpass-dir ~/data/commpass
```

**Option B: Python Script**
```python
from src.preprocessing.download_geo import GEODownloader

downloader = GEODownloader(output_dir="./data/raw")

# Download GEO datasets
print("Downloading GEO datasets...")
results = downloader.download_all()

# Process CoMMpass
print("Processing CoMMpass...")
expr, clinical, treatment = downloader.download_commpass(
    commpass_dir="~/data/commpass"
)

if expr is not None:
    print(f"✓ Loaded {expr.shape[0]} genes × {expr.shape[1]} samples")
    print(f"✓ Loaded clinical data for {len(clinical)} patients")
    print(f"✓ Loaded {len(treatment)} treatment records")
```

## Output Files

After processing, data is saved to:

```
./data/raw/
├── GSE19784/
│   ├── GSE19784_expression.parquet
│   ├── GSE19784_phenotype.csv
│   └── GSE19784_metadata.json
├── GSE39754/
│   └── ...
├── GSE2658/
│   └── ...
└── MMRF_CoMMpass_IA21/
    ├── MMRF_CoMMpass_IA21_expression.parquet
    ├── MMRF_CoMMpass_IA21_clinical.csv
    ├── MMRF_CoMMpass_IA21_treatment.csv
    └── MMRF_CoMMpass_IA21_metadata.json
```

## Understanding the Files

### Expression Matrix
- **Format:** Parquet (compressed, efficient)
- **Dimensions:** Genes × Samples
- **Index:** Gene IDs (Entrez Gene)
- **Values:** TPM (transcripts per million) normalized counts

### Clinical Data
Standardized to match GEO datasets with columns:
- `OS`: Overall survival time (months)
- `OS_event`: Death event (0/1)
- `PFS`: Progression-free survival (months)
- `PFS_event`: Progression event (0/1)
- `ISS`: International Staging System (1-3)
- `Cytogenetics`: Genetic aberrations
- `dataset_id`: Always "MMRF_CoMMpass_IA21"

### Treatment Response
Raw data with:
- Treatment types and regimens
- Response classifications (CR, PR, MR, NC, PD)
- Outcome indicators
- Timing information

## Retry Logic (Automatic)

All downloads automatically retry on network failures:
- **3 total attempts** with exponential backoff
- **Delays:** 5s → 10s → 20s between attempts
- **Covers:** Connection errors, timeouts, HTTP errors
- **Logging:** Detailed messages about each retry

No configuration needed - just works automatically!

## Troubleshooting

### Missing CoMMpass Instructions
If you need to see download instructions again:
```python
from src.preprocessing.download_geo import GEODownloader
GEODownloader()._print_commpass_instructions()
```

### File Not Found Error
Check that all 3 required files are in the directory:
```bash
ls -la ~/data/commpass/
# Should show:
# MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv
# MMRF_CoMMpass_IA21_PER_PATIENT.tsv
# MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv
```

### Network Timeouts
The pipeline will automatically retry 3 times. If it still fails:
1. Check internet connection
2. Check MMRF portal status
3. Try downloading manually and specify directory

### Data Format Issues
Ensure downloaded files are:
- Tab-separated (`.tsv`)
- UTF-8 encoding
- Not compressed (not `.gz`, `.zip`)

## Data Privacy & Citation

**Important:**
- CoMMpass data is protected by MMRF Data Use Agreement
- Individual-level data cannot be shared publicly
- Publication results must cite: "MMRF CoMMpass IA21 dataset"
- Contact MMRF for publication approval

## Support

- MMRF Researcher Gateway: https://research.themmrf.org/contact
- Pipeline documentation: See `DATA_ACQUISITION_FIXES.md`

## Next Steps

After successfully loading data:

1. **Quality Control:**
   ```python
   expr_df, clinical_df, _ = downloader.download_commpass(
       commpass_dir="~/data/commpass"
   )
   print(f"Genes: {expr_df.shape[0]}")
   print(f"Samples: {expr_df.shape[1]}")
   print(f"Clinical variables: {clinical_df.shape[1]}")
   ```

2. **Merge with GEO Data:**
   - All datasets are standardized for consistent analysis
   - OS/PFS/ISS columns are normalized across datasets
   - Combine expression matrices by common genes

3. **Downstream Analysis:**
   - Survival analysis (OS/PFS by molecular subtypes)
   - Treatment response prediction
   - Gene expression signatures
   - Cross-dataset validation

Enjoy robust, reproducible data acquisition!
