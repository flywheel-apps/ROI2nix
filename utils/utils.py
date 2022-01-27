"""
 This module represents functionality used by the `run.py` script for readability
 and encapsulation.  References to `prior art` are given where appropriate.
 Areas for future implementation are noted for ease of prospective implementation.
"""

import logging
import os
import os.path as op
import re
import shutil
from ast import literal_eval as leval
from collections import OrderedDict
from pathlib import Path
from zipfile import ZipFile
import glob
from scipy import stats


import dicom2nifti
import nibabel as nib
import nrrd
import numpy as np
import pydicom
import requests
from skimage import draw
from dicom2nifti.image_volume import SliceOrientation


log = logging.getLogger(__name__)


class InvalidDICOMFile(Exception):
    """Exception raised when an invalid DICOM file is encountered.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

class InvalidConversion(Exception):
    """Exception raised when a conversion cannot be done (nifti to dicom, nifti to nrrd).

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

class InvalidROIError(Exception):
    """Exception raised when session ROI data is invalid.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message


def io_proxy_wado(
    api_key, api_key_prefix, project_id, study=None, series=None, instance=None
):
    """
    Request wrapper for io-proxy api (https://{instance}/io-proxy/docs#/).

    This offers a complete indexing of the dicoms in the wado database.

    Admittedly, there could be other ways to do this.

    Args:
        api_key (str): Full instance api-key
        api_key_prefix (str): Type of user (e.g. 'scitran-user')
        project_id (str): Project ID to inquire
        study (str, optional): DICOM StudyUID. Defaults to None.
        series (str, optional): DICOM SeriesUID. Defaults to None.
        instance (str, optional): DICOM InstanceUID. Defaults to None.

    Returns:
        dict/list: A dictionary for dicom tags or a list of dictionaries with dicom tags
    """
    base_url = Path(api_key.split(":")[0])
    base_url /= "io-proxy/wado"
    base_url /= "projects/" + project_id
    if study:
        base_url /= "studies/" + study
    if series:
        base_url /= "series/" + series
        base_url /= "instances"
    if instance:
        base_url /= instance
        base_url /= "tags"
    base_url = "https://" + str(base_url)

    headers = {
        "Authorization": api_key_prefix + " " + api_key,
        "accept": "application/json",
    }

    req = requests.get(base_url, headers=headers)

    return leval(req.text)


def convert_ROI_to_nifti_form(
    fw_client, project_id, file_obj, imagePath, ohifViewer_info
):
    """
    Converts ohifViewer ROI from a dicom to that of an individual nifti format.

    Dicom annotations occur on the session-level info and with UID-centric formatting.
    This function places the annotations in the file_obj info with file-centric
    formatting.

    Args:
        fw_client (flywheel.Client): The active flywheel client
        project_id (str): The id of the project to search for io-proxy elements
        file_obj (flywheel.File): File object data
        imagePath (str): The DICOM representation of the ohifViewer ROI imagePath
        ohifViewer_info (dict): The full session-level ohifViewer ROI dictionary.

    Returns:
        dict: File object (file_obj) dictionary
        str: Returns perpendicular axis to the plane (perp_char, e.g. "x")
    """
    host = fw_client._fw.api_client.configuration.host[:-8]
    api_key_prefix = fw_client._fw.api_client.configuration.api_key_prefix[
        "Authorization"
    ]
    api_key_hash = fw_client._fw.api_client.configuration.api_key["Authorization"]
    api_key = ":".join([host.split("//")[1], api_key_hash])

    out_ohifViewer_info = {"measurements": {}}

    study_id, series_id = imagePath.split("$$$")[:2]

    # Grab metadata of all instances related to this study and series.
    # "Could" open all the dicoms in the series to get this... but it is in a database
    # so why not grab it from the database.
    instances = io_proxy_wado(api_key, api_key_prefix, project_id, study_id, series_id)

    for roi_type in ["FreehandRoi", "RectangleRoi", "EllipticalRoi"]:
        if ohifViewer_info["measurements"].get(roi_type):
            for roi in ohifViewer_info["measurements"].get(roi_type):
                if imagePath in roi["imagePath"]:
                    # grab the instance (slice) that roi is drawn on from the imagePath
                    instance_id = roi["imagePath"].split("$$$")[2]
                    # get the stored instance (SOP UID) from instances list
                    slice_instance = [
                        i for i in instances if i["00080018"]["Value"][0] == instance_id
                    ][0]
                    # (0020, 0013) Instance Number
                    # Integer "voxel" coordinate direction of the perpendicular axis.
                    InstanceNumber = slice_instance["00200013"]["Value"][0]

                    if roi_type not in out_ohifViewer_info["measurements"].keys():
                        out_ohifViewer_info["measurements"][roi_type] = []

                    # (0020, 0037) Image Orientation Patient
                    # Needed to determine axis perpendicular to slice
                    ImageOrientationPatient = list(
                        map(round, slice_instance["00200037"]["Value"])
                    )
                    # from https://stackoverflow.com/questions/34782409/understanding-dicom-image-attributes-to-get-axial-coronal-sagittal-cuts
                    # https://groups.google.com/g/comp.protocols.dicom/c/GW04ODKR7Tc?pli=1
                    if ImageOrientationPatient == [1, 0, 0, 0, 1, 0]:
                        # Axial Slices
                        perp_char = "z"
                    elif ImageOrientationPatient == [0, 1, 0, 0, 0, -1]:
                        # Sagittal Slices
                        perp_char = "x"
                    elif ImageOrientationPatient == [1, 0, 0, 0, 0, -1]:
                        # Coronal Slices
                        perp_char = "y"

                    # InstanceNumber is 1-indexed, niftis are 0-indexed.
                    roi[
                        "imagePath"
                    ] = f"dicom.nii.gz#{perp_char}-{InstanceNumber-1},t-0$$$0"
                    out_ohifViewer_info["measurements"][roi_type].append(roi)

    file_obj["info"].update({"ohifViewer": out_ohifViewer_info})

    return file_obj, perp_char


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """
    poly2mask converts polygon vertex coordinates into a filled polygon

    Origin of this code was in a scikit-image issue:
        https://github.com/scikit-image/scikit-image/issues/1103#issuecomment-52378754

    Args:
        vertex_row_coords (list): x-coordinates of the vertices
        vertex_col_coords (list): y-coordinates of the vertices
        shape (tuple): The size of the two-dimensional array to fill the
            polygon

    Returns:
        numpy.Array: Two-Dimensional numpy array of boolean values: True
            for within the polygon, False for outside the polygon
    """
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape
    )
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def freehand2mask(roi_points, shape, axes_flips, swap_axes=False):
    """
    Create a binary mask for the polygon described in roi_points.

    Args:
        roi_points (list): Points representing vertices of the freehand polygon.
        shape (tuple): The size of the two-dimensional array to fill.
        axes_flips (tuple): Indicates if the affine flips each axes, x or y.
        swap_axes (bool, optional): If the x and y axes need to be swapped.
            Defaults to False.

    Returns:
        numpy.Array: Two-Dimensional numpy array of boolean values: True
            for within the polygon, False for outside the polygon
    """
    x_flip, y_flip = axes_flips

    # Initialize x,y coordinates for each polygonal point
    X = []
    Y = []
    if isinstance(roi_points, list):
        for h in roi_points:
            if x_flip:  # orientation_char == "x":
                X.append(shape[0] - h["x"])
            else:
                X.append(h["x"])
            if y_flip:
                Y.append(shape[1] - h["y"])
            else:
                Y.append(h["y"])

    # We loop back to the original point to form a closed polygon
    X.append(X[0])
    Y.append(Y[0])

    # If these coordinates need to be swapped
    if swap_axes:
        Z = X
        X = Y
        Y = Z

    # If this slice already has data (i.e. this label was used in an ROI
    # perpendicular to the current slice) we need to have the logical or
    # of that data and the new data
    return poly2mask(X, Y, shape)


def rectangle2mask(start, end, shape, axes_flips, swap_axes=False):
    """
    rectangle2mask converts rectangle coordinates into a two-dimensional mask

    Args:
        start (tuple): Upper left coordinate of bounding box
        end (tuple): Lower right coordinate of bounding box
        shape (tuple): The size of the two-dimensional array to fill
        axes_flips (tuple): Indicates if the affine flips each axes, x or y.
        swap_axes (bool, optional): If the x and y axes need to be swapped.
            Defaults to False.

    Returns:
        numpy.Array: Two-Dimensional numpy array of boolean values: True
            for within the rectangle, False for outside the rectangle
    """

    x_flip, y_flip = axes_flips

    if x_flip:
        start["x"] = shape[0] - start["x"]
        end["x"] = shape[0] - end["x"]

    if y_flip:
        start["y"] = shape[1] - start["y"]
        end["y"] = shape[1] - end["y"]

    # Convert bounding box into the clockwise-rendered coordinates of a rectangle
    vertex_row_coords = [start["x"], end["x"], end["x"], start["x"], start["x"]]
    vertex_col_coords = [start["y"], start["y"], end["y"], end["y"], start["y"]]
    # If these coordinates need to be swapped
    if swap_axes:
        vertex_swp_coords = vertex_row_coords
        vertex_row_coords = vertex_col_coords
        vertex_col_coords = vertex_swp_coords
    # Pass to poly2mask
    return poly2mask(vertex_row_coords, vertex_col_coords, shape)


def ellipse2mask(start, end, shape, axes_flips, swap_axes=False):
    """
    ellipse2mask converts ellipse parameters into a two-dimensional mask

    Args:
        start (tuple): Upper left coordinate of bounding box
        end (tuple): Lower right coordinate of bounding box
        shape (tuple): The size of the two-dimensional array to fill
        axes_flips (tuple): Indicates if the affine flips each axes, x or y.
        swap_axes (bool, optional): If the x and y axes need to be swapped.
            Defaults to False.

    Returns:
        numpy.Array: Two-Dimensional numpy array of boolean values: True
            for within the ellipse, False for outside the ellipse
    """

    x_flip, y_flip = axes_flips

    if x_flip:
        start["x"] = shape[0] - start["x"]
        end["x"] = shape[0] - end["x"]

    if y_flip:
        start["y"] = shape[1] - start["y"]
        end["y"] = shape[1] - end["y"]

    r_radius, c_radius = ((end["x"] - start["x"]) / 2, (end["y"] - start["y"]) / 2)
    r_center, c_center = (start["x"] + r_radius, start["y"] + c_radius)

    if swap_axes:
        r_radius, c_radius = c_radius, r_radius
        r_center, c_center = c_center, r_center

    fill_row_coords, fill_col_coords = draw.ellipse(
        r_center, c_center, r_radius, c_radius, shape
    )

    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True

    return mask


def output_ROI_info(context, labels):
    """
    output_ROI_info [summary]

    Args:
        context ([type]): [description]
        labels ([type]): [description]
    """
    if len(labels) > 0:
        lines = []
        lines.append("label,index,voxels,volume (mm^3)\n")
        for label in labels:
            index = labels[label].index
            voxels = labels[label].num_voxels
            volume = labels[label].volume
            lines.append("{},{},{},{}\n".format(label, index, voxels, volume))
        csv_file = open(op.join(context.output_dir, "ROI_info.csv"), "w")
        csv_file.writelines(lines)
        csv_file.close()
    else:
        log.warning("There were no labels to process.")


def write_3D_Slicer_CTBL(context, file_input, labels):
    """
    write_3D_Slicer_CTBL saves the label information to a 3D Slicer colortable
        file format.

    Args:
        context (flywheel.gear_context.GearContext): gear context for output_dir
        file_input (flywheel.models.file.File): Input file with metadata
        labels (OrderedDict): Ordered dictionary of ROI label attributes
    """
    if len(labels) > 0:
        ctbl = open(
            op.join(
                context.output_dir,
                "ROI_ALL_labels_" + file_input["location"]["name"][:-7] + ".ctbl",
            ),
            "w",
        )
        for label in labels:
            ctbl.write(
                "{} ".format(labels[label].index)
                + "{} ".format(label)
                + "{} ".format(labels[label].RGB[0])
                + "{} ".format(labels[label].RGB[1])
                + "{} ".format(labels[label].RGB[2])
                + "255\n"
            )

        ctbl.close()
