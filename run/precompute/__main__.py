def main():
  import os
  from . import _constants as constants
  from ._dataset import download_file
  from ._process_data import process


  for data_id in constants.DATASET:
    download_file(
      f"datasets/{data_id}/matrix.hdf5",
      os.path.join(constants.DATA_FOLDER, f"{data_id}.hdf5")
    )
    process(data_id)


if __name__ == "__main__":
  main()
