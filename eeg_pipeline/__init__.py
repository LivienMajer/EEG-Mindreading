"""
EEG QC + preprocessing for the mind-reading hackathon.

    python -m eeg_pipeline --data_dir data --out_dir prep       # run everything

    from eeg_pipeline import load_preprocessed
    d = load_preprocessed('prep', 1)
"""
from .auto import run_pipeline
from .preprocessing import (load_preprocessed, load_test, load_test_labels, load_train,
                         prepare_subject)

__all__ = ['run_pipeline', 'load_preprocessed', 'prepare_subject',
           'load_train', 'load_test', 'load_test_labels']
