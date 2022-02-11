from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from zipfile import ZipFile
import os
import shutil
import logging
import pydicom
from utils.utils import InvalidConversion, InvalidDICOMFile, InvalidROIError
from flywheel import Client, FileEntry
from collections import OrderedDict
import glob
import numpy as np
import utils.utils as utils
import re
import subprocess as sp
from utils.Labels import RoiLabel
from scipy import stats

log = logging.getLogger(__name__)

"""
Collectors - 
Process 2 of 4
These are designed to pull roi information from flywheel.
The ROI information can be on the session level (for dicoms) or the file level (niftis)
They must identify where the information is, and then in the case of
session level dicom data, they will only pull out ROI's that are associated with
the input file provided to the gear. 

Responsibilities:
1. locate ohif metadata
2. isolate ROI's associated with input file
3. return curated metadata
    
Full process:
1. Prep
2. Collect
3. Generate
4. Convert

"""


@dataclass
class LabelCollector:
    fw_client: Client
    orig_dir: Path
    file_object: FileEntry
    collector: CollWorker = None

    def __post_init__(self):
        self.collector = self.collector(self.fw_client, self.file_object)

    def collect_rois(self):
        self.collector.collect()


class CollWorker(ABC):
    def __init__(self, fw_client, file_object, orig_dir):
        self.fw_client = fw_client
        self.orig_dir = orig_dir
        self.file_object = file_object
        self.ohifviewer_info = {}
        self.validROIs = ["RectangleRoi", "EllipticalRoi", "FreehandRoi"]

    def get_ohif_info(self):
        # Can assume file is reloaded already as the file object does this automatically
        flywheel_file = self.file_object.flywheel_file
        if flywheel_file.get("info") and flywheel_file["info"].get("ohifViewer"):
            self.ohifviewer_info = flywheel_file["info"].get("ohifViewer")

        else:
            # session stores the OHIF annotations
            session = self.fw_client.get_session(flywheel_file["parents"]["session"])
            self.ohifviewer_info = session.info.get("ohifViewer")

        if not self.ohifViewer_info:
            error_message = "Session info is missing ROI data for selected DICOM file."
            raise InvalidROIError(error_message)

    @abstractmethod
    def collect(self):
        pass


class DicomRoiCollector(CollWorker):
    def collect(self):
        self.get_ohif_info()
        studyUID, seriesUID = self.get_current_study_series_uid()
        self.identify_rois_on_image(studyUID, seriesUID)

        return self.ohifviewer_info

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

    def identify_rois_on_image(self, studyInstanceUid, seriesInstanceUid):
        imagePath = f"{studyInstanceUid}$$${seriesInstanceUid}$$$"
        # check for imagePath in ohifViewer info
        if imagePath not in str(self.ohifviewer_info):
            error_message = "Session info is missing ROI data for selected DICOM file."
            raise InvalidROIError(error_message)

        new_ohif_measurements = {}
        for measurement_type, measurements in self.ohifviewer_info.get(
            "measurements", {}
        ).items():

            # Ensure this is an ROI type we can use
            if measurement_type not in self.validROIs:
                log.info(f"Measurement type {measurement_type} invalid, skipping")
                continue

            current_measurements = []

            # This could be dict comprehension but I'm leaving it broken out for readability
            # We're going to look through each measurement and extract those that
            # are specifically on this particular dicom, identified by its series instance uid
            for roi in measurements:
                if roi.get("SeriesInstanceUID") == self.seriesInstanceUid:
                    current_measurements.append(roi)

            new_ohif_measurements[measurement_type] = current_measurements

        # Now the info here only has ROI's related to this particular dicom.
        self.ohifviewer_info = new_ohif_measurements

class NiftiRoiCollector(CollWorker):
    pass