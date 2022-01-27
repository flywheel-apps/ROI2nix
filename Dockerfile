# Dockerfile exported by GearBuilderGUI.Stash edits before export again

# Inheriting from established docker image:
FROM ubuntu:focal

# Inheriting from established docker image:
LABEL maintainer="Flywheel <support@flywheel.io>"
ENV DEBIAN_FRONTEND noninteractive

# Install APT dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \ 
    python3-setuptools \
    libgdcm-tools \
    curl \
    libarchive-tools \
    cmake \
    make \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV CXX=/usr/bin/gcc
ENV DCMCOMMIT=003f0d19f1e57b0129c9dcf3e653f51ca3559028
RUN curl -#L  https://github.com/rordenlab/dcm2niix/archive/$DCMCOMMIT.zip | bsdtar -xf- -C /usr/local
WORKDIR /usr/local/dcm2niix-${DCMCOMMIT}/build
RUN cmake -DUSE_OPENJPEG=ON -MY_DEBUG_GE=ON -DCMAKE_CXX_COMPILER=/usr/bin/c++ ../ && \
    make && \
    make install


# Make directory for flywheel spec (v0):
ENV FLYWHEEL /flywheel/v0
WORKDIR ${FLYWHEEL}

# Install PIP Dependencies
COPY requirements.txt ${FLYWHEEL}/requirements.txt
RUN pip3 install --upgrade pip && \ 
    pip3 install -r requirements.txt && \
    rm -rf /root/.cache/pip

# Specify ENV Variables
ENV \ 
    PATH=$PATH  \ 
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH 

# Copy executable/manifest to Gear
COPY run.py manifest.json my_tests.py ${FLYWHEEL}/
ADD utils ${FLYWHEEL}/utils
RUN chmod a+x /flywheel/v0/run.py

# ENV preservation for Flywheel Engine
RUN python3 -c 'import os, json; f = open("/tmp/gear_environ.json", "w");json.dump(dict(os.environ), f)'

# Configure entrypoint
ENTRYPOINT ["/flywheel/v0/run.py"]