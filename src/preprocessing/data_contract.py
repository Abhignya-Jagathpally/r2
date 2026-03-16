"""
Data Contract & Preprocessing Verification
============================================
FROZEN preprocessing contract (Karpathy autoresearch pattern).
Schema validation, column dtypes, value ranges.
Hash-based verification that preprocessing code hasn't changed.

Author: PhD Researcher 2
Date: 2026
"""

import logging
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DataContract:
    """Frozen data contract for reproducible preprocessing."""

    def __init__(self, contract_id: Optional[str] = None):
        """
        Initialize data contract.

        Parameters
        ----------
        contract_id : Optional[str]
            Unique contract identifier. If None, generated from timestamp.
        """
        self.contract_id = contract_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.schema = {}
        self.dtypes = {}
        self.value_ranges = {}
        self.checksums = {}
        self.is_frozen = False
        self.created_at = datetime.now().isoformat()

    def define_schema(
        self,
        columns: List[str],
        index_name: Optional[str] = None,
        index_dtype: Optional[str] = None,
    ) -> None:
        """
        Define expected schema.

        Parameters
        ----------
        columns : List[str]
            Expected column names.
        index_name : Optional[str]
            Index name.
        index_dtype : Optional[str]
            Index dtype.
        """
        if self.is_frozen:
            raise RuntimeError("Contract is frozen. Cannot modify.")

        self.schema = {
            "columns": columns,
            "index_name": index_name,
            "index_dtype": index_dtype,
            "n_columns": len(columns),
        }
        logger.info(f"Schema defined: {len(columns)} columns")

    def define_dtypes(self, dtype_map: Dict[str, str]) -> None:
        """
        Define expected column data types.

        Parameters
        ----------
        dtype_map : Dict[str, str]
            Column name → dtype string (e.g., "float32", "int32").
        """
        if self.is_frozen:
            raise RuntimeError("Contract is frozen. Cannot modify.")

        self.dtypes = dtype_map
        logger.info(f"Dtypes defined for {len(dtype_map)} columns")

    def define_value_ranges(self, range_map: Dict[str, Tuple[float, float]]) -> None:
        """
        Define expected value ranges.

        Parameters
        ----------
        range_map : Dict[str, Tuple[float, float]]
            Column name → (min, max) tuple.
        """
        if self.is_frozen:
            raise RuntimeError("Contract is frozen. Cannot modify.")

        self.value_ranges = range_map
        logger.info(f"Value ranges defined for {len(range_map)} columns")

    def freeze(self) -> None:
        """Freeze contract (prevent further modifications)."""
        self.is_frozen = True
        logger.info(f"Contract {self.contract_id} frozen at {datetime.now().isoformat()}")

    def save(self, output_path: Path) -> None:
        """
        Serialize contract to disk.

        Parameters
        ----------
        output_path : Path
            Output pickle file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Contract saved: {output_path}")

    @staticmethod
    def load(input_path: Path) -> "DataContract":
        """
        Load contract from disk.

        Parameters
        ----------
        input_path : Path
            Input pickle file path.

        Returns
        -------
        DataContract
            Loaded contract.
        """
        with open(input_path, "rb") as f:
            contract = pickle.load(f)
        logger.info(f"Contract loaded: {input_path} (ID: {contract.contract_id})")
        return contract

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert contract to dictionary.

        Returns
        -------
        Dict[str, Any]
            Contract as dictionary.
        """
        return {
            "contract_id": self.contract_id,
            "created_at": self.created_at,
            "is_frozen": self.is_frozen,
            "schema": self.schema,
            "dtypes": self.dtypes,
            "value_ranges": self.value_ranges,
            "checksums": self.checksums,
        }


class DataValidator:
    """Validate data against contract."""

    def __init__(self, contract: DataContract):
        """
        Initialize validator.

        Parameters
        ----------
        contract : DataContract
            Frozen contract to validate against.
        """
        self.contract = contract
        self.validation_report = {}

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema against contract.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        logger.info("Validating schema...")
        errors = []

        if "columns" not in self.contract.schema:
            errors.append("Contract has no schema defined.")
            return False, errors

        expected_columns = set(self.contract.schema["columns"])
        actual_columns = set(df.columns)

        missing_cols = expected_columns - actual_columns
        extra_cols = actual_columns - expected_columns

        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            errors.append(f"Extra columns: {extra_cols}")

        if self.contract.schema.get("index_name"):
            if df.index.name != self.contract.schema["index_name"]:
                errors.append(f"Index name mismatch: expected {self.contract.schema['index_name']}, "
                             f"got {df.index.name}")

        is_valid = len(errors) == 0
        logger.info(f"Schema validation: {'PASS' if is_valid else 'FAIL'}")
        return is_valid, errors

    def validate_dtypes(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame column dtypes.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        logger.info("Validating dtypes...")
        errors = []

        if not self.contract.dtypes:
            logger.warning("No dtypes defined in contract.")
            return True, []

        for col, expected_dtype in self.contract.dtypes.items():
            if col not in df.columns:
                continue

            actual_dtype = str(df[col].dtype)
            if not self._dtype_matches(actual_dtype, expected_dtype):
                errors.append(f"Column {col}: expected {expected_dtype}, got {actual_dtype}")

        is_valid = len(errors) == 0
        logger.info(f"Dtype validation: {'PASS' if is_valid else 'FAIL'}")
        return is_valid, errors

    def validate_value_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame value ranges.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        logger.info("Validating value ranges...")
        errors = []

        if not self.contract.value_ranges:
            logger.warning("No value ranges defined in contract.")
            return True, []

        for col, (min_val, max_val) in self.contract.value_ranges.items():
            if col not in df.columns:
                continue

            col_min = df[col].min()
            col_max = df[col].max()

            if col_min < min_val:
                errors.append(f"Column {col}: min value {col_min} < {min_val}")
            if col_max > max_val:
                errors.append(f"Column {col}: max value {col_max} > {max_val}")

        is_valid = len(errors) == 0
        logger.info(f"Value range validation: {'PASS' if is_valid else 'FAIL'}")
        return is_valid, errors

    def validate_no_missing(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that no data is missing.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        logger.info("Validating missing data...")
        errors = []

        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            errors.append(f"Found {missing_count} missing values")

        is_valid = len(errors) == 0
        logger.info(f"Missing data validation: {'PASS' if is_valid else 'FAIL'}")
        return is_valid, errors

    def validate_all(self, df: pd.DataFrame) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Run all validations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        Dict[str, Tuple[bool, List[str]]]
            Results of all validation checks.
        """
        logger.info(f"Running full validation for contract {self.contract.contract_id}...")

        results = {
            "schema": self.validate_schema(df),
            "dtypes": self.validate_dtypes(df),
            "value_ranges": self.validate_value_ranges(df),
            "no_missing": self.validate_no_missing(df),
        }

        all_valid = all(is_valid for is_valid, _ in results.values())
        logger.info(f"Overall validation: {'PASS' if all_valid else 'FAIL'}")

        self.validation_report = results
        return results

    def _dtype_matches(self, actual: str, expected: str) -> bool:
        """Check if dtype strings match (with flexibility for platform differences)."""
        # Normalize dtype names
        actual_norm = actual.replace("int", "int").replace("float", "float").lower()
        expected_norm = expected.replace("int", "int").replace("float", "float").lower()

        return actual_norm == expected_norm or actual == expected

    def report(self) -> str:
        """
        Generate validation report string.

        Returns
        -------
        str
            Human-readable validation report.
        """
        report_lines = [
            f"Data Validation Report (Contract: {self.contract.contract_id})",
            f"Timestamp: {datetime.now().isoformat()}",
            "=" * 60,
        ]

        for check_name, (is_valid, errors) in self.validation_report.items():
            status = "PASS" if is_valid else "FAIL"
            report_lines.append(f"\n{check_name.upper()}: {status}")
            if errors:
                for error in errors:
                    report_lines.append(f"  - {error}")

        return "\n".join(report_lines)


class PreprocessingCodeHasher:
    """Hash-based verification of preprocessing code integrity."""

    @staticmethod
    def hash_file(file_path: Path) -> str:
        """
        Compute SHA256 hash of a file.

        Parameters
        ----------
        file_path : Path
            Path to file.

        Returns
        -------
        str
            Hex-encoded SHA256 hash.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def hash_module(module_path: Path) -> Dict[str, str]:
        """
        Compute hashes for all Python files in a module.

        Parameters
        ----------
        module_path : Path
            Path to module directory.

        Returns
        -------
        Dict[str, str]
            Mapping of file name → SHA256 hash.
        """
        hashes = {}
        for py_file in module_path.glob("*.py"):
            hashes[py_file.name] = PreprocessingCodeHasher.hash_file(py_file)
        return hashes

    @staticmethod
    def verify_module_integrity(module_path: Path, expected_hashes: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Verify that preprocessing code hasn't been modified.

        Parameters
        ----------
        module_path : Path
            Path to module directory.
        expected_hashes : Dict[str, str]
            Expected file hashes from contract.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, modified_files)
        """
        actual_hashes = PreprocessingCodeHasher.hash_module(module_path)
        modified_files = []

        for file_name, expected_hash in expected_hashes.items():
            if file_name not in actual_hashes:
                modified_files.append(f"MISSING: {file_name}")
            elif actual_hashes[file_name] != expected_hash:
                modified_files.append(f"MODIFIED: {file_name}")

        for file_name in actual_hashes:
            if file_name not in expected_hashes:
                modified_files.append(f"ADDED: {file_name}")

        is_valid = len(modified_files) == 0
        return is_valid, modified_files


def create_expression_contract(n_samples: int, n_genes: int) -> DataContract:
    """
    Create standard contract for gene expression data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_genes : int
        Number of genes.

    Returns
    -------
    DataContract
        Initialized contract.
    """
    contract = DataContract()

    # Schema
    columns = [f"gene_{i}" for i in range(n_genes)]
    contract.define_schema(columns=columns, index_name="sample_id")

    # Dtypes: all float32 for expression
    contract.define_dtypes({col: "float32" for col in columns})

    # Value ranges: typically log2-normalized [-5, 15]
    contract.define_value_ranges({col: (-5.0, 15.0) for col in columns})

    contract.freeze()
    return contract


def create_pathway_contract(n_samples: int, n_pathways: int) -> DataContract:
    """
    Create standard contract for pathway scores.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_pathways : int
        Number of pathways.

    Returns
    -------
    DataContract
        Initialized contract.
    """
    contract = DataContract()

    # Schema
    columns = [f"pathway_{i}" for i in range(n_pathways)]
    contract.define_schema(columns=columns, index_name="sample_id")

    # Dtypes: all float32 for pathway scores
    contract.define_dtypes({col: "float32" for col in columns})

    # Value ranges: pathway scores typically [-3, 3]
    contract.define_value_ranges({col: (-3.0, 3.0) for col in columns})

    contract.freeze()
    return contract


def main():
    """Example usage."""
    logger.info("Data contract system initialized.")


if __name__ == "__main__":
    main()
