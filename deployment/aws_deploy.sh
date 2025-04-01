#!/bin/bash
# Deploy to AWS EC2
docker build -t ecg-model .
aws ecr create-repository --repository-name ecg-model
docker tag ecg-model:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/ecg-model:latest
aws ecr get-login-password | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ecg-model:latest
aws ec2 run-instances --image-id ami-0abcdef1234567890 --count 1 --instance-type t2.micro --key-name MyKeyPair