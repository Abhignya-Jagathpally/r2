"""
Unit Tests for Preprocessing Pipeline
======================================
Tests with synthetic data for:
- Probe mapping
- Normalization
- Pathway scoring
- Quality control
- Data validation

Author: PhD Researcher 2
Date: 2026
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.probe_mapping import ProbeMapper
from src.preprocessing.normalization import ExpressionNormalizer, NormalizationContract
from src.preprocessing.pathway_scoring import PathwayScorer
from src.preprocessing.quality_control import QualityController
from src.preprocessing.data_contract import DataContract, DataValidator, PreprocessingCodeHasher
from src.preprocessing.harmonization import PathwayHarmonizer


class TestProbeMapping(unittest.TestCase):
    """Test probe-to-gene mapping."""

    def setUp(self):
        """Set up test fixtures."""
        self.mapper = ProbeMapper(species="human")

        # Create synthetic expression matrix (probes × samples)
        np.random.seed(42)
        n_probes = 100
        n_samples = 20

        self.probe_ids = [f"AFFX_{i:05d}" for i in range(n_probes)]
        self.sample_ids = [f"sample_{i:02d}" for i in range(n_samples)]

        expr_matrix = np.random.normal(5, 2, (n_probes, n_samples))
        self.expr_df = pd.DataFrame(
            expr_matrix,
            index=self.probe_ids,
            columns=self.sample_ids,
        )
        self.expr_df.index.name = "probe_id"

    def test_mapping_creates_output(self):
        """Test that mapping returns a DataFrame."""
        gene_expr, stats = self.mapper.map_probes_to_genes(
            self.expr_df,
            platform="GPL570"
        )

        self.assertIsInstance(gene_expr, pd.DataFrame)
        self.assertGreater(gene_expr.shape[0], 0)
        self.assertEqual(gene_expr.shape[1], self.expr_df.shape[1])

    def test_mapping_stats_reported(self):
        """Test that mapping statistics are reported."""
        gene_expr, stats = self.mapper.map_probes_to_genes(
            self.expr_df,
            platform="GPL570"
        )

        self.assertIn("total_probes", stats)
        self.assertIn("mapped_probes", stats)
        self.assertIn("unique_genes", stats)
        self.assertEqual(stats["total_probes"], len(self.probe_ids))

    def test_validation_on_mapped_genes(self):
        """Test validation of mapped gene expression."""
        gene_expr, stats = self.mapper.map_probes_to_genes(
            self.expr_df,
            platform="GPL570"
        )

        validation = self.mapper.validate_mapping(gene_expr)

        self.assertIn("n_genes", validation)
        self.assertIn("missing_values", validation)
        self.assertEqual(validation["n_genes"], gene_expr.shape[0])


class TestNormalization(unittest.TestCase):
    """Test expression normalization."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_genes = 500
        n_samples = 30

        gene_ids = [f"ENSG_{i:05d}" for i in range(n_genes)]
        sample_ids = [f"sample_{i:02d}" for i in range(n_samples)]

        # Create expression with platform bias
        expr_matrix = np.random.normal(6, 1.5, (n_genes, n_samples))
        # Add sample-wise batch effect
        for i in range(n_samples):
            expr_matrix[:, i] += np.random.normal(i * 0.5, 0.1)

        self.expr_df = pd.DataFrame(
            expr_matrix,
            index=gene_ids,
            columns=sample_ids,
        )
        self.expr_df.index.name = "gene_id"

    def test_quantile_normalization(self):
        """Test quantile normalization."""
        contract = NormalizationContract()
        normalizer = ExpressionNormalizer(contract=contract)

        norm_expr, params = normalizer.quantile_normalize_array(self.expr_df)

        # Check output shape
        self.assertEqual(norm_expr.shape, self.expr_df.shape)

        # Check that means are more similar across samples
        orig_sample_means = self.expr_df.mean(axis=0)
        norm_sample_means = norm_expr.mean(axis=0)

        orig_cv = orig_sample_means.std() / orig_sample_means.mean()
        norm_cv = norm_sample_means.std() / norm_sample_means.mean()

        self.assertLess(norm_cv, orig_cv)

    def test_low_expression_filtering(self):
        """Test low-expression gene filtering."""
        contract = NormalizationContract()
        normalizer = ExpressionNormalizer(contract=contract)

        filt_expr, stats = normalizer.low_expression_filter(
            self.expr_df,
            percentile=50
        )

        self.assertGreater(filt_expr.shape[0], 0)
        self.assertLessEqual(filt_expr.shape[0], self.expr_df.shape[0])
        self.assertIn("retained_genes", stats)

    def test_log2_transformation(self):
        """Test log2 transformation."""
        contract = NormalizationContract()
        normalizer = ExpressionNormalizer(contract=contract)

        log_expr = normalizer.log2_transform(self.expr_df, pseudocount=0.0)

        # Check that values are log-transformed
        self.assertLess(log_expr.mean().mean(), self.expr_df.mean().mean())
        self.assertEqual(log_expr.shape, self.expr_df.shape)

    def test_normalization_contract_frozen(self):
        """Test that contract becomes frozen."""
        contract = NormalizationContract()
        normalizer = ExpressionNormalizer(contract=contract)

        self.assertFalse(contract.is_frozen)

        # Run pipeline
        norm_expr, stats = normalizer.normalize_pipeline(
            self.expr_df,
            platform_type="array"
        )

        # Contract should be frozen
        self.assertTrue(contract.is_frozen)

        # Attempting to modify should raise error
        with self.assertRaises(RuntimeError):
            contract.add_params(foo="bar")


class TestPathwayScoring(unittest.TestCase):
    """Test pathway-level scoring."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_genes = 1000
        n_samples = 50

        gene_ids = [f"GENE_{i:04d}" for i in range(n_genes)]
        sample_ids = [f"sample_{i:02d}" for i in range(n_samples)]

        expr_matrix = np.random.normal(0, 1, (n_genes, n_samples))
        self.expr_df = pd.DataFrame(
            expr_matrix,
            index=gene_ids,
            columns=sample_ids,
        )
        self.expr_df.index.name = "gene_symbol"

    def test_pathway_scorer_initialization(self):
        """Test scorer initialization."""
        scorer = PathwayScorer(method="ssgsea")
        self.assertEqual(scorer.method, "ssgsea")

    def test_load_curated_pathways(self):
        """Test loading of curated MM pathways."""
        scorer = PathwayScorer()
        pathways = scorer.load_curated_mm_pathways()

        self.assertIsInstance(pathways, dict)
        self.assertGreater(len(pathways), 0)
        self.assertIn("proliferation", pathways)
        self.assertIn("nfkb_signaling", pathways)

    def test_filter_pathways_by_genes(self):
        """Test filtering pathways to expressed genes."""
        scorer = PathwayScorer()
        test_pathways = {
            "pathway_1": ["GENE_0000", "GENE_0001", "GENE_0002"],
            "pathway_2": ["GENE_9999", "UNKNOWN_GENE"],  # One unknown
            "pathway_3": ["UNKNOWN_1", "UNKNOWN_2"],  # All unknown
        }

        filtered = scorer.filter_pathways_by_genes(self.expr_df, test_pathways)

        # Should keep pathway_1 (3 genes present)
        # and pathway_2 (1 gene present, but < 3)
        # and remove pathway_3 (0 genes)
        self.assertIn("pathway_1", filtered)
        self.assertEqual(len(filtered["pathway_1"]), 3)

    def test_pathway_scoring_output_shape(self):
        """Test pathway scoring output shape."""
        scorer = PathwayScorer(method="ssgsea")

        test_pathways = {
            f"pathway_{i}": [f"GENE_{j:04d}" for j in range(i*10, (i+1)*10)]
            for i in range(10)
        }

        # Filter
        filtered = scorer.filter_pathways_by_genes(self.expr_df, test_pathways)

        # Score
        pathway_scores = scorer.score_with_ssgsea(self.expr_df, filtered)

        # Check output
        self.assertEqual(pathway_scores.shape[1], self.expr_df.shape[1])  # Samples as columns
        self.assertGreater(pathway_scores.shape[0], 0)  # Some pathways


class TestQualityControl(unittest.TestCase):
    """Test quality control functions."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
        feature_ids = [f"feature_{i:02d}" for i in range(n_features)]

        expr_matrix = np.random.normal(0, 1, (n_samples, n_features))
        self.expr_df = pd.DataFrame(
            expr_matrix,
            index=sample_ids,
            columns=feature_ids,
        )

    def test_pca_outlier_detection(self):
        """Test PCA-based outlier detection."""
        qc = QualityController()

        pca_scores, outlier_flags, stats = qc.detect_pca_outliers(
            self.expr_df,
            n_pcs=10,
            threshold_sd=3.0
        )

        self.assertEqual(len(outlier_flags), self.expr_df.shape[0])
        self.assertIn("n_outliers", stats)
        self.assertIn("outlier_rate", stats)

    def test_missing_data_analysis(self):
        """Test missing data analysis."""
        qc = QualityController()

        # Add some missing values
        self.expr_df.iloc[0:5, 0:3] = np.nan

        missing_stats = qc.analyze_missing_data(self.expr_df)

        self.assertIn("total_missing", missing_stats)
        self.assertGreater(missing_stats["total_missing"], 0)

    def test_qc_report_generation(self):
        """Test QC report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = QualityController(output_dir=Path(tmpdir))

            report_path = qc.generate_qc_report(
                self.expr_df,
                dataset_id="test_dataset"
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(report_path.suffix == ".html")


class TestDataContract(unittest.TestCase):
    """Test data contract system."""

    def setUp(self):
        """Set up test fixtures."""
        self.contract = DataContract(contract_id="test_contract_001")

    def test_contract_definition(self):
        """Test defining contract schema."""
        columns = ["gene_1", "gene_2", "gene_3"]
        self.contract.define_schema(columns=columns, index_name="sample_id")

        self.assertEqual(self.contract.schema["n_columns"], 3)
        self.assertEqual(self.contract.schema["index_name"], "sample_id")

    def test_contract_freezing(self):
        """Test contract freezing."""
        self.assertFalse(self.contract.is_frozen)

        self.contract.freeze()

        self.assertTrue(self.contract.is_frozen)

        with self.assertRaises(RuntimeError):
            self.contract.define_schema(columns=["a", "b"])

    def test_contract_serialization(self):
        """Test contract serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.contract.define_schema(columns=["g1", "g2"])
            self.contract.freeze()

            path = Path(tmpdir) / "contract.pkl"
            self.contract.save(path)

            loaded = DataContract.load(path)

            self.assertEqual(loaded.contract_id, self.contract.contract_id)
            self.assertTrue(loaded.is_frozen)

    def test_data_validation(self):
        """Test data validation against contract."""
        # Create contract
        self.contract.define_schema(
            columns=["gene_1", "gene_2"],
            index_name="sample_id"
        )
        self.contract.define_dtypes({
            "gene_1": "float32",
            "gene_2": "float32",
        })
        self.contract.freeze()

        # Create valid dataframe
        valid_df = pd.DataFrame(
            np.random.randn(10, 2).astype("float32"),
            columns=["gene_1", "gene_2"],
        )
        valid_df.index.name = "sample_id"

        # Validate
        validator = DataValidator(self.contract)
        results = validator.validate_all(valid_df)

        self.assertTrue(results["schema"][0])
        self.assertTrue(results["dtypes"][0])

    def test_data_validation_fails_on_mismatch(self):
        """Test validation failure on schema mismatch."""
        # Create contract
        self.contract.define_schema(columns=["gene_1", "gene_2"])
        self.contract.freeze()

        # Create mismatched dataframe
        invalid_df = pd.DataFrame(
            np.random.randn(10, 3),
            columns=["gene_1", "gene_2", "gene_3"],
        )

        validator = DataValidator(self.contract)
        results = validator.validate_all(invalid_df)

        # Schema validation should fail
        self.assertFalse(results["schema"][0])


class TestPathwayHarmonization(unittest.TestCase):
    """Test cross-study harmonization."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        self.pathways_study1 = pd.DataFrame(
            np.random.randn(20, 100),
            columns=[f"pathway_{i:02d}" for i in range(100)],
        )
        self.pathways_study2 = pd.DataFrame(
            np.random.randn(25, 95),
            columns=[f"pathway_{i:02d}" for i in range(5, 100)],
        )
        self.pathways_study3 = pd.DataFrame(
            np.random.randn(30, 90),
            columns=[f"pathway_{i:02d}" for i in range(10, 100)],
        )

    def test_harmonizer_initialization(self):
        """Test harmonizer initialization."""
        harmonizer = PathwayHarmonizer()
        self.assertIsInstance(harmonizer, PathwayHarmonizer)

    def test_identify_common_pathways(self):
        """Test identification of common pathways."""
        harmonizer = PathwayHarmonizer()

        study_pathways = {
            "study1": self.pathways_study1,
            "study2": self.pathways_study2,
            "study3": self.pathways_study3,
        }

        common, study_specific = harmonizer.identify_common_pathways(study_pathways)

        self.assertGreater(len(common), 0)
        self.assertEqual(len(study_specific), 3)

    def test_standardize_scales(self):
        """Test pathway score standardization."""
        harmonizer = PathwayHarmonizer()

        study_pathways = {
            "study1": self.pathways_study1,
            "study2": self.pathways_study2,
        }

        standardized = harmonizer.standardize_pathway_scales(
            study_pathways,
            method="zscore"
        )

        # Check that means are close to 0
        for study_id, df in standardized.items():
            self.assertLess(abs(df.mean().mean()), 0.1)

    def test_harmonized_matrix_creation(self):
        """Test harmonized matrix creation."""
        harmonizer = PathwayHarmonizer()

        study_pathways = {
            "study1": self.pathways_study1,
            "study2": self.pathways_study2,
        }

        harmonized_pathways, harmonized_metadata = harmonizer.create_harmonized_matrix(
            study_pathways
        )

        # Should have all samples
        self.assertEqual(
            harmonized_pathways.shape[0],
            self.pathways_study1.shape[0] + self.pathways_study2.shape[0]
        )

        # Should only have common pathways
        self.assertGreater(harmonized_pathways.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
