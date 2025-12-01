import pytest
import main

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_import_main():
    assert main is not None

def test_dataset_load():
    # load_dataset() handles missing files gracefully
    loaded = main.load_dataset()
    assert loaded in [True, False]
