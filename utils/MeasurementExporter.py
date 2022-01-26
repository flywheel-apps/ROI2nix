from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from utils.utils import InvalidConversion, InvalidDICOMFile, InvalidROIError
from zipfile import ZipFile
import os
import shutil
import logging
import pydicom
from collections import OrderedDict
import glob
import numpy as np
import utils.utils as utils
import re
import subprocess as sp

log = logging.getLogger(__name__)


class MeasurementExport(ABC):
    """Exports measurements from files on flywheel"""

    def __init__(
        self,
        fw_client,
        input_file_path,
        orig_file_type,
        dest_file_type,
        combine,
        file_object,
        work_dir,
        output_dir,
    ):
        self.input_file_path = Path(input_file_path)
        self.identifier = {}
        self.orig_file_type = orig_file_type
        self.dest_file_type = dest_file_type
        self.fw_client = fw_client
        self.combine = combine
        self.ohifViewer_info = {}
        self.file_object = file_object
        self.validROIs = ["RectangleRoi", "EllipticalRoi", "FreehandRoi"]
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.dtype = np.uint8
        self.bits = 8

        if (orig_file_type, dest_file_type) in [("nifti", "nrrd"), ("nifti", "dicom")]:
            raise InvalidConversion(f"Cannot convert from {orig_file_type} to {dest_file_type}")

        self.class_setup()

    @abstractmethod
    def class_setup(self):
        """additional class setup stuff"""
        pass

    @abstractmethod
    def prep_data(self):
        """does and prepwork needed to the data"""
        pass

    @abstractmethod
    def make_data(self):
        pass

    @abstractmethod
    def get_labels(self):
        """Gets all the labels associated with this particular dicom"""
        pass

    @abstractmethod
    def save_data(self):
        """saves the data"""
        pass

    def process_file(self):
        self.prep_data()
        self.get_labels()
        self.make_data()


class MeasurementExportFromDicom(MeasurementExport):
    def class_setup(self):

        self.dicom_files = (
            []
        )  # A list of all the paths to individual dicom files.  Matches list order of self.dicoms
        self.dicoms = []  # Will be a list of opened dicom files (read by pydicom)
        self.orig_dicom_dir = (
            None  # will be the working directory that dicoms are extracted/coppied to
        )
        self.studyInstanceUid = None  # study instance uid of the current dicom
        self.seriesInstanceUid = None  # series instance uid of the current dicom
        self.output_dicom_dir = None  # ouput directory for converted dicoms
        self.labels = []  # A list of all measurement label names for this dicom
        self.shape = ()  # The shape of the data matrix that will have ROI's put on it.

        self.project_id = self.file_object.parents["project"]
        # Prioritize dicom file-level ROI annotations
        #   -- if they should occur this way soon...
        #   -- dicom annotations are currently on session-level info.
        if self.file_obj.get("info") and self.file_obj["info"].get("ohifViewer"):
            self.ohifViewer_info = self.file_obj["info"].get("ohifViewer")

        else:
            # session stores the OHIF annotations
            session = self.fw_client.get(self.file_object.parents["session"])
            self.ohifViewer_info = session.info.get("ohifViewer")

        if not self.ohifViewer_info:
            error_message = "Session info is missing ROI data for selected DICOM file."
            raise InvalidROIError(error_message)

    def prep_data(self):
        # need studyInstanceUid and seriesInstanceUid from DICOM series to select
        # appropriate records from the Session-level OHIF viewer annotations:
        # e.g. session.info.ohifViewer.measurements.EllipticalRoi[0].imagePath =
        #   studyInstanceUid$$$seriesInstanceUid$$$sopInstanceUid$$$0

        # if archived, unzip dicom into work/dicom/
        self.orig_dicom_dir = self.work_dir / "dicom"
        self.orig_dicom_dir.mkdir(parents=True, exist_ok=True)
        if self.input_file_path.endswith(".zip"):
            # Unzip, pulling any nested files to a top directory.
            dicom_zip = ZipFile(self.input_file_path)
            for member in dicom_zip.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
                # copy file (taken from zipfile's extract)
                source = dicom_zip.open(member)
                target = open(os.path.join(self.orig_dicom_dir, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
            # dicom_zip.extractall(path=dicom_dir)
        else:
            shutil.copy(self.input_file_path, self.orig_dicom_dir)

        # open one dicom file to extract studyInstanceUid and seriesInstanceUid
        # If this was guaranteed to be a part of the dicom-file metadata, we could grab it
        # from there. No guarantees. But all the tags are in the WADO database...
        dicom = None
        for root, _, files in os.walk(str(self.orig_dicom_dir), topdown=False):
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
            self.studyInstanceUid = dicom.StudyInstanceUID
            self.seriesInstanceUid = dicom.SeriesInstanceUID
        else:
            error_message = "An invalid dicom file was encountered."
            raise InvalidDICOMFile(error_message)

        self.output_dicom_dir = self.work_dir / "converted_dicom"
        # convert dicom to nifti (work/dicom.nii.gz)
        try:
            # Copy dicoms to a new directory
            if not os.path.exists(self.output_dicom_dir):
                shutil.copytree(self.orig_dicom_dir, self.output_dicom_dir)
        except Exception as e:
            log.exception(e)
            error_message = (
                "An invalid dicom file was encountered. "
                '"SeriesInstanceUID", "InstanceNumber", '
                '"ImageOrientationPatient", or "ImagePositionPatient"'
                " may be missing from the DICOM series."
            )
            raise InvalidDICOMFile(error_message)

        imagePath = f"{self.studyInstanceUid}$$${self.seriesInstanceUid}$$$"
        # check for imagePath in ohifViewer info
        if imagePath not in str(self.ohifViewer_info):
            error_message = "Session info is missing ROI data for selected DICOM file."
            raise InvalidROIError(error_message)

        new_ohif_measurements = {}
        for measurement_type, measurements in self.ohifViewer_info.get(
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
        self.ohifViewer_info = new_ohif_measurements

    def make_data(self):

        len_labels = len(self.labels)
        if len_labels > 0:
            log.info("Found %s ROI labels", len_labels)
        else:
            log.error("Found NO ROI labels")

        self.get_dicoms()

        # Check save_single_ROIs
        # TODO: There should also be an extra section here for binary vs bitmask.  Currently only binary is implemented
        for label in self.labels:
            data = self.label2data(label)
            # data *= self.labels[label]["index"]
            data *= 1
            self.convert_working_dir(data)
            self.save_working_dir(label)

        if self.combine:
            data = np.zeros(self.shape, dtype=self.dtype)
            for label in self.labels:
                label_data = self.label2data(label)
                label_data *= self.labels[label]["index"]
                data += label_data

            self.convert_working_dir(data)
            self.save_working_dir(combined=True)

    def get_labels(self):
        """
        gather_ROI_info extracts label-name along with bitmasked index and RGBA
            color code for each distinct label in the ROI collection.

        Args:
            file_obj (flywheel.models.file.File): The flywheel file-object with
                ROI data contained within the `.info.ROI` metadata.

        Returns:
            OrderedDict: the label object populated with ROI attributes
        """

        # dictionary for labels, index, R, G, B, A
        labels = OrderedDict()

        # React OHIF Viewer
        roi_list = [
            individual_value
            for roi_list in self.ohifViewer_info.values()
            for individual_value in roi_list
        ]

        for roi in roi_list:
            if roi.get("location"):
                if roi["location"] not in labels.keys():
                    labels[roi["location"]] = {
                        "index": int(2 ** (len(labels))),
                        # Colors are not yet defined so just use this
                        "color": "fbbc05",
                        "RGB": [int("fbbc05"[i : i + 2], 16) for i in [1, 3, 5]],
                    }
            else:
                log.warning(
                    "There is an ROI without a label. To include this ROI in the "
                    "output, please attach a label."
                )

        if self.combine:
            if self.dest_file_type == "dicom":
                max_labels = 31
            else:
                max_labels = 63

            if len(labels) < 8:
                self.dtype = np.uint8
                self.bits - 8
            elif len(labels) < 16:
                self.dtype = np.uint16
                self.bits = 16
            elif len(labels) < 32:
                self.dtype = np.uint32
                self.bits = 32

            elif len(labels) > max_labels:
                log.warning(
                    f"Due to the maximum integer length ({max_labels+1} bits), we can "
                    f"only keep track of a maximum of {max_labels} ROIs with a bitmasked "
                    f"combination. You have {len(labels)} ROIs."
                )

        self.labels = labels

    def get_dicoms(self):
        # Acquire ROI data
        globdir = self.orig_dicom_dir / "*.dcm"
        dicom_files = glob.glob(globdir.as_posix())
        dicom_files.sort()
        dicom_files = [Path(d) for d in dicom_files]
        dicoms = [pydicom.read_file(d) for d in dicom_files]
        example_dicom = dicoms[0]
        self.shape = [
            example_dicom.pixel_array.shape[0],
            example_dicom.pixel_array.shape[1],
            len(dicom_files),
        ]
        self.dicoms = dicoms
        self.dicom_files = dicom_files

    def label2data(self, label):
        data = np.zeros(self.shape, dtype=np.bool)

        for roi_type in self.ohifViewer_info:
            for roi_list in self.ohifViewer_info[roi_type]:
                for roi in roi_list:
                    if roi.get("location") == label:
                        data = self.fill_roi_dicom_slice(
                            data,
                            roi["SOPInstanceUID"],
                            roi["handles"],
                            dicoms=self.dicoms,
                            roi_type=roi_type,
                        )

        return data

    def convert_working_dir(self, data):

        data = data.astype(self.dtype)

        for i, dicom_info in enumerate(zip(self.dicom_files, self.dicoms)):
            dicom_file = dicom_info[0]
            dicom = dicom_info[1]
            file_name = dicom_file.name
            dicom_out = self.output_dicom_dir / file_name

            arr = data[:, :, i]
            arr = arr.squeeze()

            dicom.BitsAllocated = self.bits
            dicom.BitsStored = self.bits
            dicom.HighBit = self.bits - 1
            # dicom['PixelData'].VR = "OW"

            dicom.PixelData = arr.tobytes()
            dicom.save_as(dicom_out)

    def save_working_dir(self, label=None, combined=False):

        if combined:
            label_out = "ALL"
        else:
            label_out = re.sub("[^0-9a-zA-Z]+", "_", label)

        output_filename = "ROI_" + label_out + "_" + self.file_object["name"]

        if output_filename.endswith(".gz"):
            output_filename = output_filename[: -1 * len(".nii.gz")]
        elif output_filename.endswith(".nii"):
            output_filename = output_filename[: -1 * len(".nii")]
        elif output_filename.endswith(".dicom.zip"):
            output_filename = output_filename[: -1 * len(".dicom.zip")]
        elif output_filename.endswith(".zip"):
            output_filename = output_filename[: -1 * len(".zip")]

        if self.dest_file_type == "nifti":
            command = [
                "dcm2niix",
                "-o",
                self.output_dir,
                "-f",
                output_filename,
                self.output_dicom_dir,
            ]
        elif self.dest_file_type == "nrrd":
            command = [
                "dcm2niix",
                "-o",
                self.output_dir,
                "-f",
                output_filename,
                "-e",
                "y",
                self.output_dicom_dir,
            ]

        pr = sp.Popen(command)
        pr.wait()

    def copy_nii_to_output(self):
        pass

    @staticmethod
    def fill_roi_dicom_slice(
        data, sop, roi_handles, roi_type="FreehandRoi", dicoms=None, reactOHIF=True
    ):

        dicom_sops = [d.SOPInstanceUID for d in dicoms]
        slice = dicom_sops.index(sop)

        swap_axes = True
        flips = [False, False]

        orientation_slice = data[:, :, slice]

        if roi_type == "FreehandRoi":
            if reactOHIF:
                roi_points = roi_handles["points"]
            else:
                roi_points = roi_handles

            # If this slice already has data (i.e. this label was used in an ROI
            # perpendicular to the current slice) we need to have the logical or
            # of that data and the new data

            orientation_slice[:, :] = np.logical_or(
                utils.freehand2mask(
                    roi_points, orientation_slice.shape, flips, swap_axes
                ),
                orientation_slice[:, :],
            )

        elif roi_type == "RectangleRoi":
            start = roi_handles["start"]
            end = roi_handles["end"]

            # If this slice already has data (i.e. this label was used in an ROI
            # perpendicular to the current slice) we need to have the logical or
            # of that data and the new data
            orientation_slice[:, :] = np.logical_or(
                utils.rectangle2mask(
                    start, end, orientation_slice.shape, flips, swap_axes
                ),
                orientation_slice[:, :],
            )

        elif roi_type == "EllipticalRoi":
            start = roi_handles["start"]
            end = roi_handles["end"]

            # If this slice already has data (i.e. this label was used in an ROI
            # perpendicular to the current slice) we need to have the logical or
            # of that data and the new data
            orientation_slice[:, :] = np.logical_or(
                utils.ellipse2mask(
                    start, end, orientation_slice.shape, flips, swap_axes
                ),
                orientation_slice[:, :],
            )
        data[:, :, slice] = orientation_slice

        return data
