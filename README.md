# MONOCLE3 PYTHON IMPLEMENTATION


## Installation:
```bash
git clone https://github.com/bioturing/py-monocle.git
python3 -m pip install py-monocle3/.
```

## Test:
```bash
cd py-monocle3/extras/run
pip install -r requirements.txt
python3 -m precompute
python3 -m pytest test
```

## Benchmark:
```bash
cd py-monocle3/extras/benchmark
python3 prepare_datasets.py
python3 benchmark.py
```
