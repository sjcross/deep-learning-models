# Use the TensorFlow 1.15 GPU image as the base
FROM tensorflow/tensorflow:1.15.5-gpu-py3
ENV CONDA_OVERRIDE_GLIBC=2.28

# Set a working directory inside the container
WORKDIR /workspace

# Adding missing keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update

# Install system packages
RUN apt-get -y update
RUN apt-get -y install git curl

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="/root/.pixi/bin:${PATH}"

# Create directories inside the container
RUN mkdir -p /workspace/data

# Default command
CMD ["/bin/bash"]

