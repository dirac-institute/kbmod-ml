from fibad import Fibad
from kbmod_ml.data_sets.kbmod_stamps import KbmodStamps
import os

def test_basic_loading(basic_fibad_instance):
    ds = KbmodStamps(config=basic_fibad_instance.config, split="train")

    test_tensor, label = ds[0]

    assert test_tensor.shape == (1, 21, 21)
    assert label == 0 or label == 1