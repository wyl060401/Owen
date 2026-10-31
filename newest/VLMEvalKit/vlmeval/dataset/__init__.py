"""Benchmarks used by the Small evaluation workflow."""
from .image_base import img_root_map, ImageBaseDataset
from .image_mcq import ImageMCQDataset, MMMUDataset
from .image_vqa import OCRBench
from .image_yorn import ImageYORNDataset

DATASET_CLASSES = [ImageYORNDataset, ImageMCQDataset, MMMUDataset, OCRBench]
SUPPORTED_DATASETS = [name for cls in DATASET_CLASSES for name in cls.supported_datasets()]

def build_dataset(dataset_name, **kwargs):
    for cls in DATASET_CLASSES:
        if dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)
    raise ValueError(f'Unsupported Small benchmark: {dataset_name}')

def DATASET_TYPE(dataset, *, default='MCQ'):
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            return cls.TYPE
    return default

def DATASET_MODALITY(dataset, *, default='IMAGE'):
    return 'IMAGE'
