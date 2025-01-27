import os
import pytest

from fibad import Fibad

@pytest.fixture
def test_data_path():
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, "data/")

@pytest.fixture
def basic_fibad_instance(test_data_path):
    config_path = os.path.join(test_data_path, "basic_test_config.toml")
    bfi = Fibad(config_file=config_path)

    bfi.config["general"]["data_dir"] = test_data_path

    return bfi