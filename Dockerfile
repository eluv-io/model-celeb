FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential \
    && apt-get install -y ffmpeg

RUN \
   conda create -n celeb python=3.8 -y

SHELL ["conda", "run", "-n", "celeb", "/bin/bash", "-c"]

RUN \
    conda install -y cudatoolkit=10.1 cudnn=7 nccl && \
    conda install -y -c conda-forge ffmpeg-python

COPY celeb ./celeb
COPY config.yml run.py setup.py config.py .

COPY models ./models

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

RUN /opt/conda/envs/celeb/bin/pip install .

ENTRYPOINT ["/opt/conda/envs/celeb/bin/python", "run.py"]