import os


__DIRNAME__ = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(
  os.path.dirname(__DIRNAME__), "precompute", "data")
DATASET = ["E_MTAB_5061", "Darden_et_al_2021"]
