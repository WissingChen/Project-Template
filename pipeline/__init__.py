# build a pipeline
from .base import BasePipeline
from .classification import ClassificationPipeline
pipeline_fns = {"base": BasePipeline, "classification": ClassificationPipeline}