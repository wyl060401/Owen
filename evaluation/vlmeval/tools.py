"""Materialize large benchmark TSV images into local files."""
from .smp import load, dump, localize_df
from pathlib import Path

def LOCALIZE(fname, new_fname=None):
    new_fname = new_fname or str(Path(fname).with_name(Path(fname).stem + '_local.tsv'))
    dump(localize_df(load(fname), Path(fname).stem), new_fname)
    return new_fname
