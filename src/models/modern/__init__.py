"""Modern ML models for transcriptomics risk signature prediction."""

from .deepsurv import DeepSurv, DeepCoxMLP
from .pathway_autoencoder import PathwayVAE, PathwayVAEEncoder, PathwayVAEDecoder
from .domain_adversarial import DANN, FeatureExtractor, SurvivalHead, DomainDiscriminator
from .tabpfn_classifier import TabPFNRiskClassifier, convert_survival_to_risk_classification
from .training_utils import (
    CoxPartialLikelihood,
    RankingLoss,
    ConcordanceIndex,
    EarlyStopping,
    GradientClipper,
    create_cosine_scheduler,
    compute_survival_metrics,
)
from .autoresearch_agent import (
    PreprocessingContract,
    HyperparameterSpace,
    AutoresearchAgent,
    create_search_space,
)

__all__ = [
    'DeepSurv',
    'DeepCoxMLP',
    'PathwayVAE',
    'PathwayVAEEncoder',
    'PathwayVAEDecoder',
    'DANN',
    'FeatureExtractor',
    'SurvivalHead',
    'DomainDiscriminator',
    'TabPFNRiskClassifier',
    'convert_survival_to_risk_classification',
    'CoxPartialLikelihood',
    'RankingLoss',
    'ConcordanceIndex',
    'EarlyStopping',
    'GradientClipper',
    'create_cosine_scheduler',
    'compute_survival_metrics',
    'PreprocessingContract',
    'HyperparameterSpace',
    'AutoresearchAgent',
    'create_search_space',
]
