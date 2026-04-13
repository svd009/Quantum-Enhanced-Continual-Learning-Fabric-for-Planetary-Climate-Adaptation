# SageMaker Deployment

## Prerequisites

1. AWS CLI installed and configured
2. Install SageMaker SDK

```bash
pip install sagemaker boto3
```

3. Create an S3 bucket

```bash
aws s3 mb s3://your-fedclimate-bucket
```

4. Create a SageMaker execution role in AWS IAM with:
   - AmazonSageMakerFullAccess
   - AmazonS3FullAccess

## Launch Training Job

```bash
python sagemaker/deploy.py \
    --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole \
    --bucket your-fedclimate-bucket \
    --instance ml.c5.2xlarge \
    --n-years 20 \
    --num-rounds 50
```

## Instance Recommendations

| Instance | vCPUs | RAM | Use case |
|----------|-------|-----|----------|
| ml.c5.2xlarge | 8 | 16GB | Development |
| ml.c5.4xlarge | 16 | 32GB | Full training |
| ml.g4dn.xlarge | 4 | 16GB + GPU | GPU training |
| ml.g4dn.2xlarge | 8 | 32GB + GPU | Recommended |

## Output

Model artifacts saved to:
s3://your-bucket/fedclimate/output/TIMESTAMP/output/model.tar.gz

Contains:
- model.pth — trained model weights
- metrics.json — final evaluation metrics