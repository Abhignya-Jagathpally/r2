# R2 Pipeline Data Acquisition Fixes

## Overview

This document describes the enhancements made to `/sessions/sweet-stoic-cray/r2-fresh/src/preprocessing/download_geo.py` to improve reliability and functionality of GEO and CoMMpass dataset acquisition.

## Fix 1: Exponential Backoff Retry Logic

### Problem
Network failures during GEO dataset downloads could cause the entire pipeline to fail without recovery attempts. No retry mechanism was in place for transient network issues.

### Solution
Implemented a custom `retry_with_backoff()` decorator that adds exponential backoff retry logic without external dependencies (no tenacity library).

### Implementation Details

**Decorator: `retry_with_backoff()`**
- Location: Lines 46-111 in download_geo.py
- Parameters:
  - `max_retries`: 3 attempts (default)
  - `base_delay`: 5 seconds (default)
  - `max_delay`: 60 seconds cap (default)
  - `timeout`: 30 seconds per operation (default)

**Backoff Strategy:**
- Attempt 1 fails → Wait 5s
- Attempt 2 fails → Wait 10s
- Attempt 3 fails → Wait 20s
- Attempt 4 fails → Raise exception

**Error Handling:**
- Catches: `requests.RequestException`, `TimeoutError`, `ConnectionError`, `OSError`
- Non-retryable exceptions (e.g., `ValueError`) fail immediately
- Logs detailed attempt information and final failure messages
- Informative error messages include attempt number and elapsed delay

**Application:**
The decorator is applied to the new internal method `_fetch_geo_with_retry()` (lines 127-149), which is called by the public `download_geo_dataset()` method (lines 151-176).

```python
@retry_with_backoff(max_retries=3, base_delay=5.0, max_delay=60.0, timeout=30.0)
def _fetch_geo_with_retry(self, accession: str, cache_dir: str):
    geo = GEOparse.get_GEO(
        geo=accession,
        destdir=cache_dir,
        silent=False,
        keep_series_matrix=True,
    )
    return geo
```

### Usage
No changes required for users. The retry logic is automatically applied when downloading GEO datasets:

```python
downloader = GEODownloader(output_dir="./data/raw")
expr, pheno = downloader.download_and_process("GSE19784")  # Retries automatically
```

### Benefits
- Handles transient network failures gracefully
- Reduces false negatives due to temporary connectivity issues
- Configurable parameters for different timeout/retry scenarios
- Production-grade error logging for debugging

---

## Fix 2: Enhanced CoMMpass Download and Processing

### Problem
The CoMMpass dataset requires manual registration and download from the MMRF Researcher Gateway (https://research.themmrf.org/), but the original code had only placeholder functionality with minimal guidance to users. No validation of downloaded files or parsing logic existed.

### Solution
Implemented comprehensive CoMMpass handler with:
1. Step-by-step registration and download instructions
2. File validation before processing
3. Parsing and standardization of all three data types
4. Production-grade error handling and logging
5. Command-line argument support for specifying CoMMpass directory

### Implementation Details

#### New Methods in GEODownloader Class

**1. `_print_commpass_instructions()` (Lines 516-588)**
- Displays comprehensive ASCII-formatted guide
- Covers:
  - MMRF Researcher Gateway registration steps
  - Data Use Agreement (DUA) requirements
  - Exact files to download with descriptions:
    - `MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv` (gene expression)
    - `MMRF_CoMMpass_IA21_PER_PATIENT.tsv` (clinical data)
    - `MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv` (treatment response)
  - Privacy and data sharing restrictions
  - Contact information for support

**2. `_validate_commpass_files()` (Lines 590-615)**
- Validates all three required files exist in provided directory
- Returns: `(all_valid, list_of_missing_files)`
- Provides detailed logging of found/missing files

**3. `_load_commpass_expression()` (Lines 617-638)**
- Loads gene expression TSV with gene IDs as row index
- Returns: DataFrame (genes × samples)
- Handles errors gracefully with informative messages

**4. `_load_commpass_clinical()` (Lines 640-666)**
- Loads clinical metadata from PER_PATIENT file
- Standardizes columns to match GEO dataset format using existing `standardize_clinical_metadata()`
- Returns: Standardized DataFrame with OS, PFS, ISS, Cytogenetics columns
- Preserves all clinical information

**5. `_load_commpass_treatment()` (Lines 668-688)**
- Loads treatment response data
- Minimal processing to preserve raw data
- Returns: DataFrame with treatment classifications and outcomes

**6. `download_commpass()` (Lines 690-819)**
Main public method for CoMMpass processing:
- **Parameters:** `commpass_dir` (optional path to downloaded files)
- **Returns:** `(expression_df, clinical_df, treatment_df)` or `(None, None, None)` on failure
- **Workflow:**
  1. Print instructions if no directory provided
  2. Validate directory exists
  3. Validate all required files present
  4. Load all three data types with error handling
  5. Save standardized data to output directory structure:
     - `./data/raw/MMRF_CoMMpass_IA21/MMRF_CoMMpass_IA21_expression.parquet`
     - `./data/raw/MMRF_CoMMpass_IA21/MMRF_CoMMpass_IA21_clinical.csv`
     - `./data/raw/MMRF_CoMMpass_IA21/MMRF_CoMMpass_IA21_treatment.csv`
     - `./data/raw/MMRF_CoMMpass_IA21/MMRF_CoMMpass_IA21_metadata.json`
  6. Update internal metadata tracking

**7. `download_commpass_info()` (Lines 821-845)**
- Maintained for backward compatibility
- Now delegates to new `download_commpass()` method
- Deprecated: Users should use `download_commpass(commpass_dir=...)` instead

#### Updated Main Function (Lines 847-917)
- Added command-line argument parsing:
  - `--commpass-dir PATH`: Specify CoMMpass files directory
  - `--help`: Display usage information
- Structured workflow:
  1. Download all GEO datasets
  2. Process CoMMpass if directory provided
  3. Log final status

### Usage Examples

**Print download instructions:**
```python
from src.preprocessing.download_geo import GEODownloader
downloader = GEODownloader()
downloader._print_commpass_instructions()
```

**Process CoMMpass data programmatically:**
```python
downloader = GEODownloader(output_dir="./data/raw")
expr, clinical, treatment = downloader.download_commpass(
    commpass_dir="/path/to/downloaded/files"
)
```

**Via command line:**
```bash
# Download GEO datasets only
python src/preprocessing/download_geo.py

# Download GEO + process CoMMpass
python src/preprocessing/download_geo.py --commpass-dir /path/to/commpass/files

# Show help
python src/preprocessing/download_geo.py --help
```

### File Structure Expected
```
/path/to/commpass/
├── MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv
├── MMRF_CoMMpass_IA21_PER_PATIENT.tsv
└── MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv
```

### Error Handling
- Missing directory → Print instructions and return None
- Missing files → Log specific missing files and return None
- Parse errors → Log specific file/error and return None
- Save errors → Log and return None
- All paths are validated before file operations

### Benefits
1. **User Guidance:** Clear, actionable step-by-step instructions
2. **Data Validation:** Ensures correct files before processing
3. **Standardization:** Clinical data matches format of GEO datasets
4. **Flexibility:** Command-line and programmatic interfaces
5. **Logging:** Detailed progress tracking and error messages
6. **Robustness:** Graceful error handling throughout

---

## Summary of Changes

| Component | Change | Lines |
|-----------|--------|-------|
| Imports | Added `time`, `Callable`, `TypeVar`, `Any`, `wraps` | 21, 23, 29 |
| Retry Decorator | New `retry_with_backoff()` function | 46-111 |
| GEO Download | New `_fetch_geo_with_retry()` with decorator | 127-149 |
| GEO Download | Updated `download_geo_dataset()` with error handling | 151-176 |
| CoMMpass | New `_print_commpass_instructions()` | 516-588 |
| CoMMpass | New `_validate_commpass_files()` | 590-615 |
| CoMMpass | New `_load_commpass_expression()` | 617-638 |
| CoMMpass | New `_load_commpass_clinical()` | 640-666 |
| CoMMpass | New `_load_commpass_treatment()` | 668-688 |
| CoMMpass | New `download_commpass()` main method | 690-819 |
| CoMMpass | Updated `download_commpass_info()` for compatibility | 821-845 |
| Main | Updated `main()` with CLI argument parsing | 847-917 |

---

## Testing Recommendations

1. **Retry Logic Testing:**
   - Test with simulated network failures
   - Verify backoff delays are correct (5s, 10s, 20s)
   - Verify final failure raises exception after 3 retries

2. **CoMMpass Processing Testing:**
   - Create test directory with mock CoMMpass files
   - Verify file validation catches missing files
   - Verify data is correctly parsed and standardized
   - Verify output files are created in correct locations

3. **Integration Testing:**
   - Run with `--commpass-dir` argument on real data
   - Verify GEO and CoMMpass data are processed in same session
   - Check output directory structure and file formats

---

## Dependencies

No new external dependencies added. The implementation uses only:
- Standard library: `time`, `logging`, `functools`
- Already required: `pandas`, `numpy`, `requests`, `GEOparse`

---

## Notes for Future Development

1. Consider moving retry configuration to a config file for easier tuning
2. Could add progress bars using `tqdm` for better UX with large files
3. CoMMpass data merging across data types could be automated in a future update
4. Consider async/parallel downloads for multiple GEO datasets
