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

RUN /opt/conda/envs/celeb/bin/pip install .

COPY models ./models

ENTRYPOINT ["/opt/conda/envs/celeb/bin/python", "run.py"]