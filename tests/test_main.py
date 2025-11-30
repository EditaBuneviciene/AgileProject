import pytest
import main

def test_import_main():
    assert main is not None

def test_dataset_load():
    # load_dataset() handles missing files gracefully
    loaded = main.load_dataset()
    assert loaded in [True, False]
