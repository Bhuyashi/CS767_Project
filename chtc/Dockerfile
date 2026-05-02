# Study 2 (BioViL-T) image for CHTC Docker universe jobs.
# Build from repository root:
#   docker build -f chtc/Dockerfile -t YOURDOCKERHUB/cs767_study2:latest .
#   docker push YOURDOCKERHUB/cs767_study2:latest
#
# PyTorch CUDA build matches common hi-ml / CHTC GPU lab stacks (adjust tag if needed).

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code ./code

ENV PYTHONPATH=/workspace/code
