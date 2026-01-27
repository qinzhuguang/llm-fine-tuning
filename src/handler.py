import json
import os
import shutil

import runpod
import yaml
from google.cloud import storage  # GCS 上传
from google.oauth2.service_account import Credentials
from huggingface_hub._login import login

from train import train

BASE_VOLUME = os.environ.get("BASE_VOLUME", "/runpod-volume")
if not os.path.exists(BASE_VOLUME):
    os.makedirs(BASE_VOLUME)

logger = runpod.RunPodLogger()


# GCS 上传函数
async def upload_to_gcs(local_dir: str, bucket_name: str, project_id: str, credentials_json: str, gcs_path: str):
    key_info = json.loads(credentials_json)

    # 创建 credentials
    credentials = Credentials.from_service_account_info(key_info)

    # 创建 client，手动传入 credentials）
    client = storage.Client(credentials=credentials, project=project_id)

    bucket = client.bucket(bucket_name)
    uploaded_files = 0
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            # 过滤包含checkpoint-的文件夹
            if "checkpoint-" in rel_path:
                continue
            blob = bucket.blob(f"{gcs_path}/{rel_path}")
            blob.upload_from_filename(local_path)
            uploaded_files += 1
    logger.info(f"Uploaded {uploaded_files} files to gs://{bucket_name}/{gcs_path}")


# 新增：清理函数（output）
def cleanup_output(output_dir: str):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logger.info(f"Cleaned local output dir: {output_dir}")


def validate_env(logger, rp_job_id):
    vars = ["WANDB_API_KEY", "HF_TOKEN"]

    for key in vars:
        if key not in os.environ:
            logger.error(
                f"Enviornment variable {key} not found. Please set it before running the script.", job_id=rp_job_id
            )
            raise ValueError(f"Enviornment variable {key} not found. Please set it before running the script.")


def get_output_dir(run_id):
    path = f"fine-tuning/{run_id}"
    return path


async def handler(job):
    runpod_job_id = job["id"]
    inputs = job["input"]
    run_id = inputs["run_id"]
    user_id = inputs["user_id"]
    args = inputs["args"]

    # Set output directory
    output_dir = os.path.join(BASE_VOLUME, get_output_dir(run_id), user_id)
    args["output_dir"] = output_dir

    # First save args to a temporary config file
    config_path = "/workspace/test_config.yaml"

    # Add run_name and job_id to args before saving
    args["run_name"] = run_id
    args["runpod_job_id"] = runpod_job_id
    hub_model_id = args.get("hub_model_id")
    args["hub_model_id"] = None

    yaml_data = yaml.dump(args, default_flow_style=False)
    with open(config_path, "w") as file:
        file.write(yaml_data)

        # Handle credentials
    credentials = inputs["credentials"]
    os.environ["WANDB_API_KEY"] = credentials["wandb_api_key"]
    os.environ["HF_TOKEN"] = credentials["hf_token"]

    validate_env(logger, runpod_job_id)
    login(token=os.environ["HF_TOKEN"])

    logger.info("Starting Training.")
    async for result in train(config_path):  # Pass the config path instead of args
        logger.info(result)
    logger.info("Training Complete.")

    # Cleanup
    del os.environ["WANDB_API_KEY"]
    del os.environ["HF_TOKEN"]

    # ============ 新增：训练完成后上传 GCS ============
    try:
        gcs_config = inputs["gcs_config"]
        bucket_name = gcs_config["bucket_name"]
        project_id = gcs_config["project_id"]
        credentials_json = gcs_config["credentials_json"]
        gcs_finetuned_model_path = f"{gcs_config['gcs_finetuned_model_path']}"
        hub_model_id = hub_model_id.split("-")[-1]
        gcs_path = f"{gcs_finetuned_model_path}/{user_id}/{hub_model_id}"

        await upload_to_gcs(
            local_dir=output_dir,
            bucket_name=bucket_name,
            project_id=project_id,
            credentials_json=credentials_json,
            gcs_path=gcs_path,
        )

        # 可选：上传后清理（释放 Volume 空间，如果推理用别的 endpoint）
        cleanup_output(output_dir)

        # 返回 GCS 路径（你的外部服务可直接用）
        return {
            "status": "COMPLETED",
            "model_gcs_path": f"gs://{bucket_name}/{gcs_path}",
            "message": "Training complete, model uploaded to GCS and local cleaned",
        }
    except Exception as upload_error:
        logger.error(f"GCS upload failed: {str(upload_error)}")
        # 即使上传失败也返回原结果（模型仍在 Volume）
        return {"status": "COMPLETED", "message": "Training complete, but GCS upload failed"}
    # ============ 新增结束 ============


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
