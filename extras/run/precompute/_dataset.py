from pathlib import Path

import boto3
import botocore

AWS_S3_REGION = "us-west-2"
LUIN_BENCHMARK_DATA_BUCKET = "luin-benchmark-data"
CONFIG = botocore.config.Config(signature_version=botocore.UNSIGNED)


def download_file(s3_path: str, dst_path: Path):
  resource = boto3.resource("s3", region_name=AWS_S3_REGION, config=CONFIG)
  bucket = resource.Bucket(LUIN_BENCHMARK_DATA_BUCKET)
  bucket.download_file(Key=s3_path, Filename=dst_path)
  resource.meta.client.close()
