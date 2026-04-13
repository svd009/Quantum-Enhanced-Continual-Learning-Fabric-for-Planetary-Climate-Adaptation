import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--instance", default="ml.c5.2xlarge")
    parser.add_argument("--n-years", default="10")
    parser.add_argument("--num-rounds", default="20")
    parser.add_argument("--batch-size", default="64")
    parser.add_argument("--learning-rate", default="0.001")
    parser.add_argument("--ewc-lambda", default="5000.0")
    parser.add_argument("--pde-weight", default="0.1")
    return parser.parse_args()


def main():
    args = parse_args()
    sess = sagemaker.Session()
    region = boto3.Session().region_name
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    print("Launching SageMaker job: fedclimate-" + run_id)
    print("Instance:  " + args.instance)
    print("Region:    " + str(region))

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="sagemaker",
        role=args.role,
        framework_version="2.0.1",
        py_version="py310",
        instance_type=args.instance,
        instance_count=1,
        output_path="s3://" + args.bucket + "/fedclimate/output/" + run_id,
        base_job_name="fedclimate",
        hyperparameters={
            "n-years":       args.n_years,
            "num-rounds":    args.num_rounds,
            "batch-size":    args.batch_size,
            "learning-rate": args.learning_rate,
            "ewc-lambda":    args.ewc_lambda,
            "pde-weight":    args.pde_weight,
        },
        metric_definitions=[
            {"Name": "val:rmse",   "Regex": "Val RMSE: ([0-9\\.]+)"},
            {"Name": "train:loss", "Regex": "loss=([0-9\\.]+)"},
        ],
        sagemaker_session=sess,
    )

    estimator.fit(wait=False)
    print("Job launched: " + estimator.latest_training_job.name)
    print("Monitor: https://" + str(region) + ".console.aws.amazon.com/sagemaker/home#/jobs")
    print("Output:  s3://" + args.bucket + "/fedclimate/output/" + run_id)


if __name__ == "__main__":
    main()