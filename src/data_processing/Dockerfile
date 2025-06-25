# Define several build arguments
ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=11.8
ARG CUDNN_VERSION=8
ARG HDBET_COMMIT=ae16068
ARG PYTHON_VERSION=3.11.6
ARG DEBIAN_VERSION=bookworm
ARG ANTS_VERSION=v2.5.0
ARG RADIFOX_VERSION=1.0.3

# Set base images: pytorch, ants, python and debian
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime
FROM antsx/ants:${ANTS_VERSION} as ants
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION}

# Re-declare the build arguments
ARG PYTORCH_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG HDBET_COMMIT
ARG PYTHON_VERSION
ARG DEBIAN_VERSION
ARG ANTS_VERSION
ARG RADIFOX_VERSION

# Set an environment variable for the Python
ENV PYTHONUSERBASE=/opt/python

# Write the build arguments to a JSON file for reference
RUN echo -e "{\n \
    \"PYTORCH_VERSION\": \"${PYTORCH_VERSION}\"\n \
    \"CUDA_VERSION\": \"${CUDA_VERSION}\"\n \
    \"CUDNN_VERSION\": \"${CUDNN_VERSION}\"\n \
    \"HDBET_COMMIT\": \"${HDBET_COMMIT}\"\n \
    \"PYTHON_VERSION\": \"${PYTHON_VERSION}\",\n \
    \"DEBIAN_VERSION\": \"${DEBIAN_VERSION}\"\n \
    \"ANTS_VERSION\": \"${ANTS_VERSION}\"\n \
    \"RADIFOX_VERSION\": \"${RADIFOX_VERSION}\"\n \
}" > /opt/manifest.json

# Update the package list and install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install necessary library
RUN pip install --no-cache-dir \
        radifox==${RADIFOX_VERSION} \
        SimpleITK \
        numpy \
        nibabel\
        scikit-learn \
        scikit-image \
        scipy


# Set up HD-BET
RUN pip install --no-cache-dir  git+https://github.com/MIC-DKFZ/HD-BET.git@${HDBET_COMMIT}
COPY setup_hdbet.py /opt
RUN python /opt/setup_hdbet.py && \
    rm -f /opt/setup_hdbet.py


# Install intensity-normalization with antspy dependency
RUN pip install --no-cache-dir "intensity-normalization[ants]"


# Copy registration atlas to /opt/atlas
COPY atlas /opt/atlas


# Copy ants to /opt/ants
COPY --from=ants /opt/ants /opt/ants


# Set environment variables for executable paths and library paths
ENV PATH /opt/run:/opt/ants/bin:${PYTHONUSERBASE}/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/ants/lib:${LD_LIBRARY_PATH}


# Copy executable files to /opt/run/ and set appropriate permissions
COPY --chmod=755  scripts  /opt/run/
COPY --chmod=755  run-pipeline /opt/run/


# Set the default command to be executed when the container starts
ENTRYPOINT ["run-pipeline"]
