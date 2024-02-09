import os


__DIRNAME__ = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(__DIRNAME__, "data")
LOGS_FOLDER = os.path.join(DATA_FOLDER, "logs")
DATASET = ["E_MTAB_5061", "Darden_et_al_2021"]
MONOCLE_SCRIPT = os.path.join(__DIRNAME__, "monocle3.r")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
