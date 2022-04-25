# Dockerfile exported by GearBuilderGUI.Stash edits before export again

# Inheriting from established docker image:
FROM ubuntu:focal

# Inheriting from established docker image:
LABEL maintainer="Flywheel <support@flywheel.io>"
ENV DEBIAN_FRONTEND noninteractive
# Make directory for flywheel spec (v0):
ENV FLYWHEEL /flywheel/v0
WORKDIR ${FLYWHEEL}


#############################################################
## Step 0: setup directory structures         ##
#############################################################
ENV CONVERTER_DIR=${FLYWHEEL}/converters \
    SCRIPT_DIR=${CONVERTER_DIR}/scripts \
    # Setup slicer dir https://www.slicer.org/
    SLICER_DIR=${CONVERTER_DIR}/slicer \
    SLICER_DOCKER_DIR=${CONVERTER_DIR}/slicer_docker \
    # Setup Plastimatch folder: https://plastimatch.org/
    PLASTIMATCH_DIR=${CONVERTER_DIR}/plastimatch \
    # Setup dcm2niix dir: https://github.com/rordenlab/dcm2niix
    DCM2NIIX_DIR=${CONVERTER_DIR}/dcm2niix \
    # Setup dicom2nifti dir: https://github.com/icometrix/dicom2nifti
    DICOM2NIFTI_DIR=${CONVERTER_DIR}/dicom2nifti \
    # Setup main directory for dcmheat repo
    DCMHEAT_DIR=${FLYWHEEL}/dcmheat


# Create directories
RUN mkdir ${CONVERTER_DIR} ${SLICER_DIR} ${PLASTIMATCH_DIR} ${DCM2NIIX_DIR} ${DICOM2NIFTI_DIR} ${DCMHEAT_DIR} ${SCRIPT_DIR} ${SLICER_DOCKER_DIR}


#############################################################
## Step 1: Install dependencies         ##
#############################################################
### NOTE THIS ACTUALLY INCLUDES PASTIMATCH

# Largely taking from this docker file: https://github.com/QIICR/dcmheat/blob/master/docker/Dockerfile
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
    git \
    wget \
    unzip \
    build-essential \
    xutils-dev \
    default-jre \
    dcmtk \
    plastimatch \
    libpulse-mainloop-glib0 \
    qt5-default \
    xvfb \
    libxdamage1 \
    libxcomposite-dev \
    libxcursor1 \
    libxrandr2 \
    && \
    #apt-get purge -y build-essential xutils-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/ /tmp/* /var/tmp/*

#############################################################
## Step 2: Clone dcmheat repo for supporting files         ##
#############################################################

RUN git clone https://github.com/QIICR/dcmheat.git ${DCMHEAT_DIR}
# Specific files needed:
# - docker/SlicerConvert.py


#############################################################
## Step 3: Python setup using poetry         ##
#############################################################

COPY poetry.lock pyproject.toml $FLYWHEEL/
# Install PIP Dependencies
RUN pip3 install --upgrade pip && \
    pip3 install poetry && \
    rm -rf /root/.cache/pip

RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

#############################################################
## Step 4: Install dcm2niix         ##
#############################################################

# Install openjpeg for this version of dcm2niix
# This version of dcm2niix was released  Oct 7, 2021
ENV OJ_VERSION=2.4.0
ENV OPENJPEGDIR=$FLYWHEEL/openjpeg
RUN mkdir $OPENJPEGDIR
RUN curl -#L https://github.com/uclouvain/openjpeg/archive/refs/tags/v${OJ_VERSION}.zip | bsdtar -xf- -C ${OPENJPEGDIR}
RUN cd $OPENJPEGDIR/openjpeg-${OJ_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/c++ .. && \
    make && \
    make install && \
    make clean

# Install dcm2niix
ENV CXX=/usr/bin/gcc
ENV DCMCOMMIT=003f0d19f1e57b0129c9dcf3e653f51ca3559028
RUN curl -#L  https://github.com/rordenlab/dcm2niix/archive/${DCMCOMMIT}.zip | bsdtar -xf- -C ${DCM2NIIX_DIR}
WORKDIR ${DCM2NIIX_DIR}/dcm2niix-${DCMCOMMIT}/build
RUN cmake -DUSE_OPENJPEG=ON -MY_DEBUG_GE=ON -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DOpenJPEG_DIR=$OPENJPEGDIR ../ && \
    make && \
    make install



#############################################################
## Step 5: Install plastimatch         ##
#############################################################
# NOTE THIS IS NOW TAKEN CARE OF IN THE APT-GET INSTALL OF
# STEP 1.  Leaving here in case that breaks or something...

#RUN cd /tmp && \
#	git clone https://gitlab.com/plastimatch/plastimatch.git && \
#	cd plastimatch && git checkout v1.7.2 && \
#	mkdir build && cd build && \
#	cmake -DINSTALL_PREFIX=/usr .. && \
#	make && make install && \
#	cp plastimatch /usr/bin


#############################################################
## Step 6: Install SLICER         ##
#############################################################
#### This is needed to run Slicer python scripts in a headless mode...ALLEGEDLY


## Slicer 4.11
ENV SLICER_URL http://download.slicer.org/bitstream/60add706ae4540bf6a89bf98
RUN curl -v -s -L $SLICER_URL | tar xz -C /tmp && \
    mv /tmp/Slicer* ${SLICER_DIR}/slicer
ENV PATH "${SLICER_DIR}/slicer:${PATH}"

RUN mkdir /tmp/runtime-sliceruser
ENV XDG_RUNTIME_DIR=/tmp/runtime-sliceruser


#############################################################
## Step 7: Install dicom2nifti         ##
#############################################################
# NOTE: THis is a python package and is taken care of with
# Poetry...


#############################################################
## Step 8: Setup flywheel gear stuff         ##
#############################################################


# Copy executable/manifest to Gear
COPY run.py manifest.json my_tests.py ${FLYWHEEL}/
COPY utils/SlicerScripts/RunSlicerExport.py ${SCRIPT_DIR}/RunSlicerExport.py
# COPY utils/SlicerScripts/Slicer_Export.py ${SCRIPT_DIR}/Slicer_Export.py


ADD utils ${FLYWHEEL}/utils
RUN chmod a+x /flywheel/v0/run.py

RUN mkdir ${FLYWHEEL}/rosetta
RUN mkdir ${FLYWHEEL}/scrap
RUN ln -s ${SCRIPT_DIR}/SlicerConvert.py /usr/src/SlicerConvert.py
RUN chmod 7700 /tmp/runtime-sliceruser
ADD tests ${FLYWHEEL}/tests

# Configure entrypoint
ENTRYPOINT ["/flywheel/v0/run.py"]
WORKDIR ${FLYWHEEL}

# ENV preservation for Flywheel Engine
RUN python3 -c 'import os, json; f = open("/tmp/gear_environ.json", "w");json.dump(dict(os.environ), f)'
