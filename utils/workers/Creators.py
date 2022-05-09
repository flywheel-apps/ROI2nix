from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
import glob
import logging
import numpy as np
from pathlib import Path
import pydicom
import re
from scipy import stats
from fw_file.dicom import DICOMCollection


from utils.objects.Labels import RoiLabel
import utils.roi_tools as roi_tools
from utils.workers import Converters

log = logging.getLogger(__name__)

rgba_regex = ".*\((?P<R>\d+),\s+?(?P<G>\d+),\s+?(?P<B>\d+),\s+?(?P<A>\d+?\.\d+?)\)"
rgba_regex = re.compile(rgba_regex)

"""
Generators - 
Process 3 of 4

This process actually generates the ROI image.  Due to some functionality that 
this code needs to have, the generator will contain a converter.  Therefore it
is the generators responsibility to not only create the ROI images but also
CALL the converter. 

Responsibilities:
1. based on user input (binary, bitmask, combine, etc), generate images of the ROI
2. generate the "labels" object to track mask size and bit value, etc.
3. generate the name of the new file to be saved
2. call the converter


Full process:
1. Prep
2. Collect
3. Create
4. Convert

"""


class BaseCreator(ABC):
    # Type key set on each base class to identify which class to instantiate
    type_ = None

    def __init__(
        self, orig_dir, roi_dir, output_dir, base_file_name, combine, bitmask, converter
    ):
        self.orig_dir = orig_dir
        self.roi_dir = roi_dir
        self.output_dir = output_dir  # Not actually used anywhere...
        self.base_file_name = base_file_name
        self.combine = combine
        self.bitmask = bitmask
        self.converter = converter

        ## End of user defined properties
        self.shape = [0, 0, 0]
        self.max_labels = 0
        self.dtype = np.uint8
        self.bits = 8
        self.labels = None

    def get_labels(self, ohifviewer_info):
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
            individual_roi
            for roi_type_list in ohifviewer_info.values()
            for individual_roi in roi_type_list
        ]

        roi_color = "fbbc05"

        for roi in roi_list:
            if roi.get("location"):
                if roi["location"] not in labels.keys():

                    # HAHAHA HAVE FUN WITH THIS PART!
                    # ok ok kidding.  So the way flywheel stores the color of the ROI is by this metadata tag "color"
                    # The value of this key is a string, that resembles the default below, "rgba(<R>, <G>, <B>, <A>)".
                    # This part takes that string, parses out the garbage that should not be there, and extracts the
                    # RGBA values (that is, Red, Green, Blue, Alpha).
                    # I also save it as a hex value because the old code did it that way and I don't want to mess things
                    # up.
                    roi_color = roi.get("color", "rgba(187, 192, 5, 0.2)")
                    rgba = rgba_regex.match(roi_color)
                    rgba = [
                        int(rgba.group("R")),
                        int(rgba.group("G")),
                        int(rgba.group("B")),
                        float(rgba.group("A")),
                    ]

                    labels[roi["location"]] = RoiLabel(
                        index=int(2 ** (len(labels))),
                        RGB=rgba,
                        color=f"#{hex(rgba[0])[2:]}{hex(rgba[1])[2:]}{hex(rgba[2])[2:]}",
                        label=roi["location"],
                    )

            else:
                log.warning(
                    "There is an ROI without a label. To include this ROI in the "
                    "output, please attach a label."
                )
        self.labels = labels
        return labels

    @abstractmethod
    def create(self, ohifviewer_info):
        pass

    @abstractmethod
    def get_affine(self):
        pass

    @classmethod
    def factory(
        cls,
        type_: str,
        orig_dir,
        roi_dir,
        output_dir,
        base_file_name,
        combine,
        bitmask,
        converter,
    ):
        """Return an instantiated Creator."""
        for sub in cls.__subclasses__():
            if type_.lower() == sub.type_:
                return sub(
                    orig_dir,
                    roi_dir,
                    output_dir,
                    base_file_name,
                    combine,
                    bitmask,
                    converter,
                )
        raise NotImplementedError(f"File type {type_} no supported")


class DicomCreator(BaseCreator):
    type_ = "dicom"

    def __init__(
        self, orig_dir, roi_dir, output_dir, base_file_name, combine, bitmask, converter
    ):
        super().__init__(
            orig_dir, roi_dir, output_dir, base_file_name, combine, bitmask, converter
        )
        self.dicoms = {}
        self.max_labels = 31

    def create(self, ohifviewer_info):
        labels = self.get_labels(ohifviewer_info)
        self.get_dicoms()
        self.make_data(labels, ohifviewer_info)
        return labels

    def generate_name(self, label, combine):
        if combine:
            label_out = "ALL"
        else:
            label_out = re.sub("[^0-9a-zA-Z]+", "_", label)

        output_filename = "ROI_" + label_out + "_" + self.base_file_name

        if output_filename.endswith(".gz"):
            output_filename = output_filename[: -1 * len(".nii.gz")]
        elif output_filename.endswith(".nii"):
            output_filename = output_filename[: -1 * len(".nii")]
        elif output_filename.endswith(".dicom.zip"):
            output_filename = output_filename[: -1 * len(".dicom.zip")]
        elif output_filename.endswith(".zip"):
            output_filename = output_filename[: -1 * len(".zip")]

        return output_filename

    def get_dicoms(self):
        # Acquire ROI data
        self.dicoms = DICOMCollection.from_dir(self.orig_dir)
        self.shape = [
            self.dicoms.get("Rows"),
            # May be more robust to check pixel array sizes if Rows or Columns is missing, but also might want to know if size of image changes across archive.
            self.dicoms.get("Columns"),
            # To do what you had: self.dicoms[0].dataset.raw.pixel_array.shape[1]
            len(self.dicoms),
        ]

    def get_affine(self):

        sbs = self.dicoms.get("SpacingBetweenSlices")
        pixelspacing = self.dicoms[0].get("PixelSpacing")

        # positions = np.mat(
        #     [loaded_dicom.ImagePositionPatient for dpath,loaded_dicom in self.dicoms.items()]
        # )
        # zdif = np.diff(positions, axis=0)[:, 0]
        # mzdif = stats.mode(zdif).mode[0][0]
        # mzdif = np.round(mzdif, 4)
        # ds = list(self.dicoms.values())[0]
        # affine = np.mat(
        #     [
        #         [1 * mzdif, 0, 0],
        #         [0, 1 * ds.PixelSpacing[0], 0],
        #         [0, 0, 1 * ds.PixelSpacing[1]],
        #     ]
        # )

        affine = np.array(
            [
                [1 * sbs, 0, 0],
                [0, 1 * pixelspacing[0], 0],
                [0, 0, 1 * pixelspacing[1]],
            ]
        )

        return affine

    def make_data(self, labels, ohifviewer_info):
        self.set_bit_level(labels)
        len_labels = len(labels)

        if len_labels > 0:
            log.info("Found %s ROI labels", len_labels)
        else:
            log.error("Found NO ROI labels")
        # Check save_single_ROIs
        for label_name, label_object in labels.items():
            data = self.label2data(label_name, ohifviewer_info)

            label_object.num_voxels = np.count_nonzero(data)
            data = data.astype(self.dtype)
            if self.bitmask:
                data *= labels[label_name].index

            self.save_to_roi_dir(data)
            output_filename = self.generate_name(label_name, combine=False)
            self.converter.convert(output_filename)

        if self.combine:
            data = np.zeros(self.shape, dtype=self.dtype)
            for label in labels:
                label_data = self.label2data(label, ohifviewer_info)
                label_data = label_data.astype(self.dtype)
                label_data *= labels[label].index
                data += label_data

            self.save_to_roi_dir(data)
            output_filename = self.generate_name(label_name, combine=True)
            self.converter.convert(output_filename)

    def set_bit_level(self, labels):
        # If we're not combining and binary masks are ok, we don't need to bitmask, we'll leave at default 8 bit
        if self.combine or self.bitmask:
            if len(labels) < 8:
                self.dtype = np.uint8
                self.bits = 8
            elif len(labels) < 16:
                self.dtype = np.uint16
                self.bits = 16
            elif len(labels) < 32:
                self.dtype = np.uint32
                self.bits = 32

            elif len(labels) > self.max_labels:
                log.warning(
                    f"Due to the maximum integer length ({self.max_labels+1} bits), we can "
                    f"only keep track of a maximum of {self.max_labels} ROIs with a bitmasked "
                    f"combination. You have {len(labels)} ROIs."
                )

    def label2data(self, label, ohifviewer_info):
        data = np.zeros(self.shape, dtype=bool)
        for roi_type in ohifviewer_info:
            for roi in ohifviewer_info[roi_type]:
                if roi.get("location") == label:
                    data = self.fill_roi_dicom_slice(
                        data,
                        roi["SOPInstanceUID"],
                        roi["handles"],
                        dicoms=self.dicoms,
                        roi_type=roi_type,
                    )

        return data

    def save_to_roi_dir(self, data):

        data = data.astype(self.dtype)

        for dicom_info in self.dicoms:
            i = dicom_info.InstanceNumber - 1
            dicom_file = Path(dicom_info.filepath)
            dicom_data = pydicom.read_file(dicom_file)
            file_name = dicom_file.name
            dicom_out = self.roi_dir / file_name

            arr = data[:, :, i]
            arr = arr.flatten()

            dicom_data.BitsAllocated = self.bits
            dicom_data.BitsStored = self.bits
            dicom_data.HighBit = self.bits - 1
            # dicom_data['PixelData'].VR = "OW"
            dicom_data.PixelData = arr.tobytes()
            dicom_data.save_as(dicom_out)

    @staticmethod
    def fill_roi_dicom_slice(
        data, sop, roi_handles, roi_type="FreehandRoi", dicoms=None, reactOHIF=True
    ):

        dicom_sops = dicoms.bulk_get("SOPInstanceUID")
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
                roi_tools.freehand2mask(
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
                roi_tools.rectangle2mask(
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
                roi_tools.ellipse2mask(
                    start, end, orientation_slice.shape, flips, swap_axes
                ),
                orientation_slice[:, :],
            )
        data[:, :, slice] = orientation_slice

        return data


class NiftiCreator(BaseCreator):
    type_ = "nifti-notimplemented"

    def get_affine(self):
        pass

    def create(self):
        pass
