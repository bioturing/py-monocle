import os
import time
import psutil
import pandas as pd

from pathlib import Path
import multiprocessing as mp
from tempfile import TemporaryDirectory


SPAWN_CTX = mp.get_context("spawn")
__DIRNAME__ = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(__DIRNAME__, "scripts")
DATASETS = os.path.join(__DIRNAME__, "datasets")
LOGS_PATH = os.path.join(DATASETS, "logs")
STDOUT = 1
STDERR = 2

os.makedirs(LOGS_PATH, exist_ok=True)


def _get_scripts(func_name: str):
  filenames = [
      filename.split('.')
      for filename in os.listdir(SCRIPTS_PATH)
      if filename.startswith(func_name)
  ]

  return filenames


def _run_process(process: mp.Process):
  process.start()
  process_utils = psutil.Process(pid=process.pid)
  interval, cpu_mem = 0, 0

  while process.is_alive():
    cur_mem = process_utils.memory_info().rss
    cpu_mem = max(cpu_mem, cur_mem)
    time.sleep(interval)
    interval = min(0.1, interval + 0.01)

  process.join()
  process.close()
  return cpu_mem


def _redirect_exec(cmd: tuple, stdout_path: Path, stderr_path: Path):
  fdout = os.open(stdout_path, os.O_WRONLY | os.O_CREAT)
  os.dup2(fdout, STDOUT)
  fderr = os.open(stderr_path, os.O_WRONLY | os.O_CREAT)
  os.dup2(fderr, STDERR)
  os.execl(*cmd)


def _run_script(func: str, lib: str, ext: str, args: tuple):
  script_path = os.path.join(SCRIPTS_PATH, f"{func}.{lib}.{ext}")
  dataset_name = os.path.basename(os.path.dirname(args[0]))
  stdout_path = os.path.join(LOGS_PATH, f"{func}.{lib}.{dataset_name}.stdout")
  stderr_path = os.path.join(LOGS_PATH, f"{func}.{lib}.{dataset_name}.stderr")

  if ext == "r":
    cmd = ("/usr/local/bin/Rscript", script_path, script_path, *args)
  elif ext == "py":
    cmd = ("/usr/bin/python3", script_path, script_path, *args)
  else:
    raise Exception(f"Not support '{ext}' extension")

  proc = SPAWN_CTX.Process(
      target=_redirect_exec,
      args=(cmd, stdout_path, stderr_path),
  )
  cpu_mem = _run_process(proc)
  return cpu_mem


def _run_function(
    func_name: str, dataset_ids: list, args_fn: callable, output_path: Path,
    extra_info: dict = None,
):
  scripts = _get_scripts(func_name)

  collect_info = {
      "Execution time": "exec_time",
      "CPU memory": "cpu_mem",
  }

  if extra_info is not None:
    collect_info.update(extra_info)

  result = {
      info: {lib: [] for _, lib, _ in scripts}
      for info in collect_info
  }
  for i, dataset_id in enumerate(dataset_ids):
    for func, lib, ext in scripts:
      with TemporaryDirectory() as tmpdir:
        path_args, result_path = args_fn(tmpdir, dataset_id)
        cpu_mem = _run_script(func, lib, ext, path_args)
        try:
          result_df = pd.read_csv(result_path)
        except Exception as e:
          for info in collect_info:
            result[info][lib].append(str(e))
        else:
          for info in collect_info:
            if info == "CPU memory":
              result[info][lib].append(cpu_mem)
            else:
              result[info][lib].append(result_df[collect_info[info]][0])

    with pd.ExcelWriter(output_path) as writer:
      for info in collect_info:
        df = pd.DataFrame(result[info], index=dataset_ids[: i + 1])
        df.to_excel(writer, sheet_name=info, index=True)


def run_monocle(dataset_ids: list):
  def args_fn(tmpdir: Path, dataset_id: str):
    dataset_path = os.path.join(
        DATASETS, dataset_id, "matrix.hdf5")
    result_path = os.path.join(tmpdir, "result.csv")
    return (dataset_path, result_path), result_path

  benchmark_result_path = os.path.join(DATASETS, "monocle3.xlsx")
  _run_function(
      "monocle", dataset_ids, args_fn, benchmark_result_path,
  )


if __name__ == "__main__":
  datasets_info_path = os.path.join(DATASETS, "datasets.csv")
  datasets_info = pd.read_csv(datasets_info_path, index_col=0)
  datasets_info.sort_values("n_cells")
  dataset_ids = list(datasets_info.index)

  run_monocle(dataset_ids)
