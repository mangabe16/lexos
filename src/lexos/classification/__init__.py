"""__init__.py.

Unified exposed module interface exposing the refactored,
object-oriented Lexos Classification infrastructure.

Last Updated: June 22, 2026
"""

from lexos.classification.trainer import Classifier, Pipeline, SklearnClassifierPipeline
from lexos.classification.mlp_pipeline import MLPPipeline
from lexos.classification.utils import PredictionSaver, save_predictions

__all__ = [
    "Classifier",
    "Pipeline",
    "SklearnClassifierPipeline",
    "MLPPipeline",
    "PredictionSaver",
    "save_predictions",
]
