# Dockerfile exported by GearBuilderGUI.Stash edits before export again

# Inheriting from established docker image:
FROM ubuntu:xenial

# Inheriting from established docker image:
LABEL maintainer="Flywheel <support@flywheel.io>"

# Install APT dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip  \ 
    zip && \ 
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install PIP Dependencies
RUN pip3 install --upgrade pip && \ 
    pip install \
    flywheel-sdk  \ 
    nibabel  \ 
    scikit-image && \ 
    rm -rf /root/.cache/pip

# Specify ENV Variables
ENV \ 
    PATH=$PATH  \ 
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH 

# Make directory for flywheel spec (v0):
ENV FLYWHEEL /flywheel/v0
WORKDIR ${FLYWHEEL}
# Copy executable/manifest to Gear
COPY run.py ${FLYWHEEL}/run.py
COPY manifest.json ${FLYWHEEL}/manifest.json
RUN chmod a+x /flywheel/v0/run.py

# ENV preservation for Flywheel Engine
RUN python3 -c 'import os, json; f = open("/tmp/gear_environ.json", "w");json.dump(dict(os.environ), f)'

ENTRYPOINT ["/flywheel/v0/run.py"]
# Configure entrypoint