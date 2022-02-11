from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from zipfile import ZipFile
import os
import shutil
import logging
import pydicom
from utils.utils import InvalidConversion, InvalidDICOMFile, InvalidROIError

from collections import OrderedDict
import glob
import numpy as np
import utils.utils as utils
import re
import subprocess as sp
from utils.Labels import RoiLabel
from scipy import stats

log = logging.getLogger(__name__)



@dataclass
class Prepper():
    work_dir: Path
    input_file_path: Path
    prepper: PrepWorker: None

    def __post_init__(self):
        self.output_dir = self.work_dir / "roi_image"
        self.orig_dir = self.work_dir / "original_image"
        self.prepper = self.prepper(self.orig_dir, self.output_dir, self.input_file_path)

    def prep_data(self):
        self.prepper.prep()



class PrepWorker(ABC):
    def __init__(self, orig_dir, output_dir, input_file_path):
        self.orig_dir = orig_dir
        self.output_dir = output_dir
        self.input_file_path = input_file_path

    @abstractmethod
    def prep(self):
        pass


class PrepDicom(PrepWorker):

    def prep(self):
        """
        For the dicom prepper, this does the following:
        1. if the dicom is zipped, unzip to `workdir/original_image`
         - if unzipped, copy the directory
        2. make a copy of the files in `workdir/roi_image` to be processed

        Returns:

        """
        self.move_dicoms_to_workdir()

        # This is actually not needed for this prep part, it's for making the image:
        studyUID, seriesUID = self.get_current_study_series_uid()

        self.copy_dicoms_for_export()

    def move_dicoms_to_workdir(self):
        # if archived, unzip dicom into work/dicom/
        self.orig_dir.mkdir(parents=True, exist_ok=True)
        if self.input_file_path.as_posix().endswith(".zip"):
            # Unzip, pulling any nested files to a top directory.
            dicom_zip = ZipFile(self.input_file_path)
            for member in dicom_zip.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
                # copy file (taken from zipfile's extract)
                source = dicom_zip.open(member)
                target = open(os.path.join(self.orig_dir, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
            # dicom_zip.extractall(path=dicom_dir)
        else:
            shutil.copy(self.input_file_path, self.orig_dir)

    def get_current_study_series_uid(self):
        # need studyInstanceUid and seriesInstanceUid from DICOM series to select
        # appropriate records from the Session-level OHIF viewer annotations:
        # e.g. session.info.ohifViewer.measurements.EllipticalRoi[0].imagePath =
        #   studyInstanceUid$$$seriesInstanceUid$$$sopInstanceUid$$$0
        # open one dicom file to extract studyInstanceUid and seriesInstanceUid
        # If this was guaranteed to be a part of the dicom-file metadata, we could grab it
        # from there. No guarantees. But all the tags are in the WADO database...
        dicom = None
        for root, _, files in os.walk(str(self.orig_dir), topdown=False):
            for fl in files:
                try:
                    dicom_path = Path(root) / fl
                    dicom = pydicom.read_file(dicom_path, force=True)
                    break
                except Exception as e:
                    log.warning("Could not open dicom file. Trying another.")
                    log.exception(e)
                    pass
            if dicom:
                break

        if dicom:
            studyInstanceUid = dicom.StudyInstanceUID
            seriesInstanceUid = dicom.SeriesInstanceUID
        else:
            error_message = "An invalid dicom file was encountered."
            raise InvalidDICOMFile(error_message)

        return studyInstanceUid, seriesInstanceUid

    def copy_dicoms_for_export(self):
        try:
            # Copy dicoms to a new directory
            if not os.path.exists(self.output_dir):
                shutil.copytree(self.orig_dir, self.output_dir)
        except Exception as e:
            log.exception(e)
            error_message = (
                "An invalid dicom file was encountered. "
                '"SeriesInstanceUID", "InstanceNumber", '
                '"ImageOrientationPatient", or "ImagePositionPatient"'
                " may be missing from the DICOM series."
            )
            raise InvalidDICOMFile(error_message)


class PrepNifti(PrepWorker):
    def prep(self):
        pass