import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from minio import Minio
from pyspark.sql import DataFrame, SparkSession


def get_minio_client(
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    secure: Optional[bool] = None,
) -> Minio:
    """
    Create MinIO client using either explicit args or environment variables.

    Env vars:
        MINIO_ENDPOINT (default: localhost:9000)
        MINIO_ACCESS_KEY (default: minioadmin)
        MINIO_SECRET_KEY (default: minioadmin)
        MINIO_SECURE (default: false)
    """
    endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")

    if secure is None:
        secure_env = os.getenv("MINIO_SECURE", "false").lower()
        secure = secure_env in ("1", "true", "yes")

    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def ensure_bucket(client: Minio, bucket_name: str) -> None:
    """Create bucket if it doesn't exist."""
    from minio.error import S3Error

    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
    except S3Error as exc:
        # If bucket already exists and we have access, ignore
        if exc.code != "BucketAlreadyOwnedByYou":
            raise


def upload_local_file(
    client: Minio,
    bucket: str,
    local_path: str,
    object_name: str,
) -> None:
    """Upload arbitrary local file to MinIO."""
    ensure_bucket(client, bucket)
    client.fput_object(bucket_name=bucket, object_name=object_name, file_path=local_path)


def upload_spark_df_as_parquet(
    df: DataFrame,
    bucket: str,
    object_prefix: str,
    mode: str = "overwrite",
) -> None:
    """
    Save Spark DataFrame as parquet to a local temp dir and upload to MinIO
    as a folder under the given prefix.
    """
    spark: SparkSession = df.sparkSession

    tmp_dir = tempfile.mkdtemp(prefix="fs_parquet_")
    try:
        local_parquet_path = os.path.join(tmp_dir, "data")
        # Write parquet locally
        df.write.mode(mode).parquet(local_parquet_path)

        client = get_minio_client()
        ensure_bucket(client, bucket)

        # Walk through written parquet and upload all files preserving structure
        base_path = Path(local_parquet_path)
        for root, _, files in os.walk(local_parquet_path):
            for f in files:
                full_path = Path(root) / f
                rel_path = full_path.relative_to(base_path)
                object_name = f"{object_prefix.rstrip('/')}/{rel_path.as_posix()}"
                client.fput_object(
                    bucket_name=bucket,
                    object_name=object_name,
                    file_path=str(full_path),
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def make_raw_key(dataset_name: str, version: str, filename: str) -> str:
    return f"raw/{dataset_name}/{version}/{filename}"


def make_features_prefix(dataset_name: str, split: str, version: str) -> str:
    return f"features/{dataset_name}/{split}/{version}"


def make_model_prefix(model_name: str, version: str) -> str:
    return f"models/{model_name}/{version}"

