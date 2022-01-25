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


def convert_dicom_to_nifti(context, input_name):

    fw_client = context.client

    # Get configuration, acquisition, and file info
    file_input = context.get_input(input_name)

    # Need updated file information.
    file_obj = file_input["object"]
    # ASSUMPTION: DICOMs will be stored at acquisition level
    acquisition = fw_client.get(file_input["hierarchy"]["id"])
    project_id = acquisition.parents["project"]
    # Prioritize dicom file-level ROI annotations
    #   -- if they should occur this way soon...
    #   -- dicom annotations are currently on session-level info.
    if file_obj.get("info") and file_obj["info"].get("ohifViewer"):
        ohifViewer_info = file_obj["info"].get("ohifViewer")

    else:
        # session stores the OHIF annotations
        session = fw_client.get(acquisition.parents["session"])
        ohifViewer_info = session.info.get("ohifViewer")

    if not ohifViewer_info:
        error_message = "Session info is missing ROI data for selected DICOM file."
        raise InvalidROIError(error_message)

    # need studyInstanceUid and seriesInstanceUid from DICOM series to select
    # appropriate records from the Session-level OHIF viewer annotations:
    # e.g. session.info.ohifViewer.measurements.EllipticalRoi[0].imagePath =
    #   studyInstanceUid$$$seriesInstanceUid$$$sopInstanceUid$$$0

    # if archived, unzip dicom into work/dicom/
    dicom_dir = context.work_dir / "dicom"
    dicom_dir.mkdir(parents=True, exist_ok=True)
    if file_input["location"]["name"].endswith(".zip"):
        # Unzip, pulling any nested files to a top directory.
        dicom_zip = ZipFile(context.get_input_path(input_name))
        for member in dicom_zip.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue

            # copy file (taken from zipfile's extract)
            source = dicom_zip.open(member)
            target = open(os.path.join(dicom_dir, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)
        #dicom_zip.extractall(path=dicom_dir)
    else:
        shutil.copy(context.get_input_path(input_name), dicom_dir)

    # open one dicom file to extract studyInstanceUid and seriesInstanceUid
    # If this was guaranteed to be a part of the dicom-file metadata, we could grab it
    # from there. No guarantees. But all the tags are in the WADO database...
    dicom = None
    for root, _, files in os.walk(str(dicom_dir), topdown=False):
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

    dicom_output = context.output_dir / "converted_dicom"
    # convert dicom to nifti (work/dicom.nii.gz)
    try:
        # Copy dicoms to a new directory
        if not os.path.exists(dicom_output):
            shutil.copytree(dicom_dir, dicom_output)
        #nii_stats = dicom2nifti.dicom_series_to_nifti(dicom_dir, dicom_output)
    except Exception as e:
        log.exception(e)
        error_message = (
            "An invalid dicom file was encountered. "
            '"SeriesInstanceUID", "InstanceNumber", '
            '"ImageOrientationPatient", or "ImagePositionPatient"'
            " may be missing from the DICOM series."
        )
        raise InvalidDICOMFile(error_message)

    imagePath = f"{studyInstanceUid}$$${seriesInstanceUid}$$$"
    # check for imagePath in ohifViewer info
    if imagePath not in str(ohifViewer_info):
        error_message = "Session info is missing ROI data for selected DICOM file."
        raise InvalidROIError(error_message)

    # file_obj, perp_char = convert_ROI_to_nifti_form(
    #     fw_client, project_id, file_obj, imagePath, ohifViewer_info
    # )
    return dicom_output, dicom_dir, file_obj


def calculate_transformation_matrix(adjustment_matrix, affine):

    # Create an inverse of the matrix that is the closest projection onto the
    # basis unit vectors of the coordinate system of the original affine.
    # This is used to determine which axes to flip

    inv_reduced_aff = np.matmul(
        # multiply by adjustment matrix, account for dicom L/R viewer presentation
        np.linalg.inv(
            # take inverse of this unitary matrix
            np.round(
                # put "1"s in each place
                np.matmul(
                    # multiply the 3x3 matrix down to size
                    affine[:3, :3],
                    # Generate the [1/norm(),...] for each column
                    # Take the norm of the column vectors.. this is the pixel width
                    np.diag(1.0 / np.linalg.norm(affine[:3, :3], axis=0)),
                )
            )
        ),
        adjustment_matrix,
    )
    print(adjustment_matrix)
    print('inv_reduced_aff:')
    print(inv_reduced_aff)
    print(affine)

    return inv_reduced_aff


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



def fill_roi_dicom_slice(data,
                            sop,
                            roi_handles,
                            roi_type="FreehandRoi",
                            dicoms = None,
                         reactOHIF=True):

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
            freehand2mask(roi_points, orientation_slice.shape, flips, swap_axes),
            orientation_slice[:, :],
        )

    elif roi_type == "RectangleRoi":
        start = roi_handles["start"]
        end = roi_handles["end"]

        # If this slice already has data (i.e. this label was used in an ROI
        # perpendicular to the current slice) we need to have the logical or
        # of that data and the new data
        orientation_slice[:, :] = np.logical_or(
            rectangle2mask(start, end, orientation_slice.shape, flips, swap_axes),
            orientation_slice[:, :],
        )

    elif roi_type == "EllipticalRoi":
        start = roi_handles["start"]
        end = roi_handles["end"]

        # If this slice already has data (i.e. this label was used in an ROI
        # perpendicular to the current slice) we need to have the logical or
        # of that data and the new data
        orientation_slice[:, :] = np.logical_or(
            ellipse2mask(start, end, orientation_slice.shape, flips, swap_axes),
            orientation_slice[:, :],
        )
    data[:, :, slice] = orientation_slice

    return data














def fill_roi_slice(
    data, img_path, roi_handles, inv_reduced_aff, roi_type="FreehandRoi", reactOHIF=True
):
    """
    Fill the data slice consistent with the roi type and label.

    Args:
        data (numpy.narray): The result in a 3D array.
        img_path (string): Provides orientation and slice info.
        roi_handles (dict): The part of the roi dictionary that holds the point data.
        inv_reduced_aff (numpy.Array): Standard unit basis inverse of nifti affine.
        roi_type (str): Type of ROI to render. Defaults to "FreehandRoi".
        reactOHIF (bool, optional): Use React(True) or Legacy Viewer(False).
            Defaults to True.
    """

    # Find orientation [Axial, Sagittal, Coronal]
    # orientation character gives us the direction perpendicular to the
    # plane of the ROI
    orientation_char = img_path[img_path.find("#") + 1]
    log.debug(f'img_path: {img_path}')
    shape = data.shape
    # orientation_coordinate gives us the coordinate along the axis
    # perpendicular to plane of the ROI
    orientation_coordinate = int(img_path[img_path.find("#") + 3 : img_path.find(",")])

    if orientation_char == "z":
        orientation_axis = np.matmul(inv_reduced_aff, [0, 0, 1])
        # If we have a positive direction on this axis, switch direction
        if np.dot(orientation_axis, np.abs(orientation_axis)) > 0:
            orientation_coordinate = shape[2] - orientation_coordinate - 1

        x_axis = np.matmul(inv_reduced_aff, [1, 0, 0])
        y_axis = np.matmul(inv_reduced_aff, [0, 1, 0])

        x_indx = x_axis.nonzero()[0][0]
        y_indx = y_axis.nonzero()[0][0]

        # The "flip" of each axis is related to the sign of its axis coordinate
        x_flip = np.dot(x_axis, np.abs(x_axis)) < 0
        y_flip = np.dot(y_axis, np.abs(y_axis)) > 0

    elif orientation_char == "y":
        orientation_axis = np.matmul(inv_reduced_aff, [0, 1, 0])
        # If we have a positive direction on this axis, switch direction
        if np.dot(orientation_axis, np.abs(orientation_axis)) > 0:
            orientation_coordinate = shape[1] - orientation_coordinate - 1

        x_axis = np.matmul(inv_reduced_aff, [1, 0, 0])
        y_axis = np.matmul(inv_reduced_aff, [0, 0, 1])

        x_indx = x_axis.nonzero()[0][0]
        y_indx = y_axis.nonzero()[0][0]

        # The "flip" of each axis is related to the sign of its axis coordinate
        x_flip = np.dot(x_axis, np.abs(x_axis)) < 0
        y_flip = np.dot(y_axis, np.abs(y_axis)) > 0

    elif orientation_char == "x":
        orientation_axis = np.matmul(inv_reduced_aff, [1, 0, 0])
        # If we have a negative direction on this axis, switch direction
        if np.dot(orientation_axis, np.abs(orientation_axis)) < 0:
            orientation_coordinate = shape[0] - orientation_coordinate - 1

        x_axis = np.matmul(inv_reduced_aff, [0, 1, 0])
        y_axis = np.matmul(inv_reduced_aff, [0, 0, 1])

        x_indx = x_axis.nonzero()[0][0]
        y_indx = y_axis.nonzero()[0][0]

        # The "flip" of each axis is related to the sign of its axis coordinate
        x_flip = np.dot(x_axis, np.abs(x_axis)) > 0
        y_flip = np.dot(y_axis, np.abs(y_axis)) > 0

    else:
        log.warning("Orientation character not recognized.")
        orientation_axis = ""
        orientation_slice = [0, 0, 0]

    log.debug(f'Orientation char: {orientation_char}')

    if all(np.abs(orientation_axis) == [1, 0, 0]):
        orientation_slice = data[orientation_coordinate, :, :]

    elif all(np.abs(orientation_axis) == [0, 1, 0]):
        orientation_slice = data[:, orientation_coordinate, :]

    elif all(np.abs(orientation_axis) == [0, 0, 1]):
        orientation_slice = data[:, :, orientation_coordinate]

    else:
        log.warning("Orientation Axis not found.")

    swap_axes = x_indx > y_indx
    flips = [x_flip, y_flip]

    if roi_type == "FreehandRoi":
        if reactOHIF:
            roi_points = roi_handles["points"]
        else:
            roi_points = roi_handles

        # If this slice already has data (i.e. this label was used in an ROI
        # perpendicular to the current slice) we need to have the logical or
        # of that data and the new data

        orientation_slice[:, :] = np.logical_or(
            freehand2mask(roi_points, orientation_slice.shape, flips, swap_axes),
            orientation_slice[:, :],
        )

    elif roi_type == "RectangleRoi":
        start = roi_handles["start"]
        end = roi_handles["end"]

        # If this slice already has data (i.e. this label was used in an ROI
        # perpendicular to the current slice) we need to have the logical or
        # of that data and the new data
        orientation_slice[:, :] = np.logical_or(
            rectangle2mask(start, end, orientation_slice.shape, flips, swap_axes),
            orientation_slice[:, :],
        )

    elif roi_type == "EllipticalRoi":
        start = roi_handles["start"]
        end = roi_handles["end"]

        # If this slice already has data (i.e. this label was used in an ROI
        # perpendicular to the current slice) we need to have the logical or
        # of that data and the new data
        orientation_slice[:, :] = np.logical_or(
            ellipse2mask(start, end, orientation_slice.shape, flips, swap_axes),
            orientation_slice[:, :],
        )


def label2data(label, info, dicoms, shape):
    """
    label2data gives the roi data block for a nifti file with `shape` and
    `info`

    This iterates through polygons drawn in any orientation and renders the
    shape contained in the indicated slice of a numpy array. This routine does
    not yet do ellipses or rectangles.

    Args:
        label (string): The label to convert the polygon data into nifti data.
        shape (list): The shape of the nifti data (e.g. [256,256,256]).
        info (dict): The `info` dictionary object of the flywheel file object.
        inv_reduced_aff (numpy.Array): Standard unit basis inverse of nifti affine.
            Example:
            .. code-block:: python

                info = {
                    "AccessionNumber": "MN080001475",
                    "roi": [
                        {
                            "color": "#ea4335",
                            "toolType" == "freehand"
                            "handles": [
                                {
                                    "y": 44.50589746553241,
                                    "x": 70.64751958224542,
                                    ...
                                },
                                {...},
                                ...
                            ]
                        },
                        {...}
                    ],
                    ....
                }

    Returns:
        numpy.array: A numpy array of `shape` with values `True` where `label`
            has been segmented and `False` on the outside.

    """
    data = np.zeros(shape, dtype=np.bool)

    # for OHIF REACT viewer:
    if (
        "ohifViewer" in info
        and "measurements" in info["ohifViewer"]
        and (
            ("FreehandRoi" in info["ohifViewer"]["measurements"])
            or ("RectangleRoi" in info["ohifViewer"]["measurements"])
            or ("EllipticalRoi" in info["ohifViewer"]["measurements"])
        )
    ):
        for roi_type in ["FreehandRoi", "RectangleRoi", "EllipticalRoi"]:
            if info["ohifViewer"]["measurements"].get(roi_type):
                for roi in info["ohifViewer"]["measurements"].get(roi_type):
                    if roi.get("location") and roi["location"] == label:
                        data = fill_roi_dicom_slice(
                            data,
                            roi["SOPInstanceUID"],
                            roi["handles"],
                            roi_type=roi_type,
                            dicoms=dicoms,
                        )

    return data


def gather_ROI_info(file_obj):
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
    if (
        "ohifViewer" in file_obj["info"].keys()
        and "measurements" in file_obj["info"]["ohifViewer"]
    ):
        if (
            ("FreehandRoi" in file_obj["info"]["ohifViewer"]["measurements"])
            or ("RectangleRoi" in file_obj["info"]["ohifViewer"]["measurements"])
            or ("EllipticalRoi" in file_obj["info"]["ohifViewer"]["measurements"])
        ):
            roi_list = []
            for roi_type in ["FreehandRoi", "RectangleRoi", "EllipticalRoi"]:
                roi_type_list = file_obj["info"]["ohifViewer"]["measurements"].get(
                    roi_type
                )
                if roi_type_list:
                    roi_list.extend(roi_type_list)

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
    # only doing this for toolType=freehand for Meteor (legacy) OHIF Viewer
    # Deprioritizing OHIF Meteor Annotations
    # ROI2Nix will not do both at the same time
    # TODO: Deprecate OHIF Meteor functionality
    elif "roi" in file_obj["info"].keys():
        for roi in file_obj["info"]["roi"]:
            if (
                roi.get("label")
                and (roi["toolType"] == "freehand")
                and (roi["label"] not in labels.keys())
            ):
                # Only if annotation type is a polygon, then grab the
                # label, create a 2^x index for bitmasking, grab the color
                # hash (e.g. #fbbc05), and translate it into RGB
                labels[roi["label"]] = {
                    "index": int(2 ** (len(labels))),
                    "color": roi["color"],
                    "RGB": [int(roi["color"][i : i + 2], 16) for i in [1, 3, 5]],
                }
    else:
        log.warning("No ROIs were found for this image.")

    if len(labels) > 63:
        log.warning(
            "Due to the maximum integer length (64 bits), we can "
            "only keep track of a maximum of 63 ROIs with a bitmasked "
            "combination. You have %i ROIs.",
            len(labels),
        )

    return labels


def calculate_ROI_volume(labels, data, affine):
    """
    calculate_ROI_volume calculates the number of voxels and volume in each of
        the ROIs

    Args:
        labels (OrderedDict): The dictionary containing ROI info
        data (numpy.ndarray): The NIfTI data with ROI within
    """
    if len(labels) > 0:
        for _, label_dict in labels.items():
            indx = label_dict["index"]
            label_data = np.bitwise_and(data, indx)
            voxels = np.sum(label_data > 0)
            label_dict["voxels"] = voxels
            volume = voxels * np.abs(np.linalg.det(affine[:3, :3]))
            label_dict["volume"] = volume  # mm^3


def save_single_ROIs(output_dir, file_input, labels, data, affine, options, binary, save_nrrd):
    """
    Output_Single_ROIs saves single ROIs to their own file. If `binary` is
        true, we have a binary mask (0,1) for the file, else the value is
        the bitmasked (power-of-two).

    Args:
        context (flywheel.gear_context.GearContext): Gear Context
        labels (OrderedDict): Label data
        data (numpy.ndarray): NIfTI data object
        affine (numpy.ndarray): NIfTI affine
        options (dict): a list of options for the nrrd format
        binary (boolean): Whether to output as binary mask or bitmasked value
    """

    if len(labels) > 0:
        for label in labels:
            indx = labels[label]["index"]
            # If binary all values are 0/1 and type is int8.
            # otherwise, accept the designated value and max integer size
            if binary:
                modifier = indx
                int_type = np.int8
            else:
                modifier = 1
                int_type = np.int64
            export_data = np.bitwise_and(data, indx).astype(int_type) / modifier

            label_out = re.sub("[^0-9a-zA-Z]+", "_", label)
            filename = "ROI_" + label_out + "_" + file_input["location"]["name"]

            write_data_out(export_data, filename, output_dir, save_nrrd, affine=affine, options=options)

    else:
        log.warning("No ROIs were found for this image.")


def save_bitmasked_ROIs(
    output_dir, labels, file_input, data, affine, options, combined_output_size, save_nrrd
):
    """
    save_bitmasked_ROIs saves all ROIs rendered into a bitmasked NIfTI file.

    Args:
        context (flywheel.gear_context.GearContext): Gear context
        labels (OrderedDict): The label attributes.
        file_input (flywheel.models.file.File): Input File object
        data (numpy.ndarray): The nifti data object
        affine (numpy.ndarray): The nifti affine
        combined_output_size (string): numpy integer size
    """

    if combined_output_size == "int8":
        bits = 7
        np_type = np.int8
    elif combined_output_size == "int16":
        bits = 15
        np_type = np.int16
    elif combined_output_size == "int32":
        bits = 31
        np_type = np.int32
    elif combined_output_size == "int64":
        bits = 63
        np_type = np.int64

    if len(labels) > bits:
        log.warning(
            "Due to the maximum integer length (%i bits), we can "
            "only keep track of a maximum of %i ROIs with a bitmasked "
            "combination. You have %i ROIs.",
            bits + 1,
            bits,
            len(labels),
        )

    if len(labels) > 1:
        # all_labels_nii = nib.Nifti1Pair(data.astype(np_type), affine)
        filename = "ROI_ALL_" + file_input["location"]["name"]

        write_data_out(data.astype(np_type), filename, output_dir, save_nrrd, affine=affine, options=options)

    else:
        log.warning("There are not enough ROIs to save an aggregate.")

def write_data_out(data, output_filename, output_dir, save_nrrd, affine=None, options=None):

    if save_nrrd:
        log.debug('save NRRD is True')
        # If the original file_input was an uncompressed NIfTI
        # ensure compression
        if output_filename.endswith(".gz"):
            output_filename = output_filename[:-1 * len('.nii.gz')]
        elif output_filename.endswith(".nii"):
            output_filename = output_filename[:-1 * len('.nii')]
        elif output_filename.endswith(".dicom.zip"):
            output_filename = output_filename[:-1 * len('.dicom.zip')]
        elif output_filename.endswith(".zip"):
            output_filename = output_filename[:-1 * len('.zip')]

        output_filename += '.nrrd'


        log.debug(f'new output name = {output_filename}')

        save_nrrd_file(data, op.join(output_dir, output_filename), options)

    else:
        label_nii = nib.Nifti1Pair(data, affine)

        # If the original file_input was an uncompressed NIfTI
        # ensure compression
        if output_filename.endswith(".nii"):
            output_filename += ".gz"
        elif output_filename.endswith(".dicom.zip"):
            output_filename = output_filename.replace(".dicom.zip", ".nii.gz")
        elif output_filename.endswith(".zip"):
            output_filename = output_filename.replace(".zip", ".nii.gz")

        nib.save(label_nii, op.join(output_dir, output_filename))


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
            index = labels[label]["index"]
            voxels = labels[label]["voxels"]
            volume = labels[label]["volume"]
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
                "{} ".format(labels[label]["index"])
                + "{} ".format(label)
                + "{} ".format(labels[label]["RGB"][0])
                + "{} ".format(labels[label]["RGB"][1])
                + "{} ".format(labels[label]["RGB"][2])
                + "255\n"
            )

        ctbl.close()


def save_dicom_out(dicom_files, dicoms, dicom_output, data):
    for i, dicom_info in enumerate(zip(dicom_files, dicoms)):

        dicom_file = dicom_info[0]
        dicom = dicom_info[1]
        file_name = dicom_file.name
        dicom_out = dicom_output/file_name

        arr = dicom.pixel_array
        print(arr.shape)
        arr = data[:,:,i]
        print(arr.shape)
        arr=arr.squeeze()
        print(arr.shape)

        dicom.BitsAllocated = 64
        dicom.BitsStored = 64

        dicom.HighBit = 63
        dicom['PixelData'].VR="OW"
        arr = np.uint64(arr)

        dicom.PixelData=arr.tobytes()
        dicom.save_as(dicom_out)


def save_rois(output_dir, file_input, labels, data, dicom_files, dicom_output, config):


    # Output individual ROIs
    save_single_ROIs(
        output_dir, file_input, labels, data, affine, options, config["save_binary_masks"], config["save_NRRD"]
    )

    # Output all ROIs in one file, if selected
    # TODO: If we want different output styles (last written, 4D Nifti)
    # we would implement that here...with a combo box in the manifest.
    if config["save_combined_output"]:
        save_bitmasked_ROIs(
            output_dir,
            labels,
            file_input,
            data,
            affine,
            options,
            config["combined_output_size"],
            config["save_NRRD"]
        )
    pass

# Heavily relying on this for the dicom header stuff: https://github.com/chunweiliu/nrrd/blob/master/DICOM_to_NRRD.py
# And this for the nifti 2 nrrd stuff: https://github.com/YuanYuYuan/MIDP/blob/master/nifti2nrrd.py

def load_dicom_options(dicom_dir, nifti):
    image = dicom2nifti.image_volume.ImageVolume(nifti)
    affine = nifti.affine

    file_names = glob.glob(os.path.join(dicom_dir,'*'))
    file_names.sort()
    #file_names = file_names[::-1]

    loaded_dicoms = [pydicom.read_file(fn) for fn in file_names]
    positions = np.mat([ld.ImagePositionPatient for ld in loaded_dicoms])
    zdif = np.diff(positions, axis=0)[:, 0]
    mzdif = stats.mode(zdif).mode[0][0]
    mzdif = np.round(mzdif, 4)

    ds = loaded_dicoms[0]
    # https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-slice-affine
    # Calc Multi:
    flist = ds.ImageOrientationPatient
    F = np.mat([[flist[3],flist[0]],[flist[4],flist[1]],[flist[5],flist[2]]])
    # #mult = np.mat([[-1,-1],
    #                [-1,-1],
    #                [1,1]])
    #F = np.multiply(F, mult)
    # dr = ds.PixelSpacing[0]
    # dc = ds.PixelSpacing[1]
    # ss = float(ds.SpacingBetweenSlices)
    xs = np.array(F[:,1]).flatten()
    ys = np.array(F[:,0]).flatten()
    n = np.cross(list(ys), list(xs))

    N=len(loaded_dicoms)
    dcT1=loaded_dicoms[0]
    dcTN=loaded_dicoms[-1]

    T1=dcT1.ImagePositionPatient
    #T1 = np.multiply(T1, [-1, -1, 1])
    TN=dcTN.ImagePositionPatient

    # Asingle = np.array([
    #     [F[0, 0], F[0, 1], n[0]],
    #     [F[1, 0], F[1, 1], n[1]],
    #     [F[2, 0], F[2, 1], n[2]],
    #     ])

    Asingle = np.array([
        [F[0, 0], F[0, 1], n[0],T1[0]],
        [F[1, 0], F[1, 1], n[1],T1[1]],
        [F[2, 0], F[2, 1], n[2],T1[2]],
        [0, 0, 0, 1]
        ])

    affine_inverse = np.linalg.inv(Asingle)
    transformed_x = np.transpose(np.dot(affine_inverse, [[1], [0], [0], [0]]))[0]
    transformed_y = np.transpose(np.dot(affine_inverse, [[0], [1], [0], [0]]))[0]
    transformed_z = np.transpose(np.dot(affine_inverse, [[0], [0], [1], [0]]))[0]
    print(transformed_x)
    print(transformed_y)
    print(transformed_z)

    x_component, y_component, z_component = dicom2nifti.image_volume.__calc_most_likely_direction__(transformed_x, transformed_y, transformed_z)
    print(x_component, y_component, z_component)

    axial_orientation = SliceOrientation()
    axial_orientation.normal_component = z_component
    axial_orientation.x_component = x_component
    axial_orientation.x_inverted = np.sign(transformed_x[axial_orientation.x_component]) < 0
    axial_orientation.y_component = y_component
    axial_orientation.y_inverted = np.sign(transformed_y[axial_orientation.y_component]) < 0
    # Find slice orientiation for the coronal size
    # Find the index of the max component to know which component is the direction in the size
    coronal_orientation = SliceOrientation()
    coronal_orientation.normal_component = y_component
    coronal_orientation.x_component = x_component
    coronal_orientation.x_inverted = np.sign(transformed_x[coronal_orientation.x_component]) < 0
    coronal_orientation.y_component = z_component
    coronal_orientation.y_inverted = np.sign(transformed_z[coronal_orientation.y_component]) < 0
    # Find slice orientation for the sagittal size
    # Find the index of the max component to know which component is the direction in the size
    sagittal_orientation = SliceOrientation()
    sagittal_orientation.normal_component = x_component
    sagittal_orientation.x_component = y_component
    sagittal_orientation.x_inverted = np.sign(transformed_y[sagittal_orientation.x_component]) < 0
    sagittal_orientation.y_component = z_component
    sagittal_orientation.y_inverted = np.sign(transformed_z[sagittal_orientation.y_component]) < 0
    # Assert that the slice normals are not equal
    print(sagittal_orientation.normal_component)
    print(coronal_orientation.normal_component)
    print(axial_orientation.normal_component)


    new_affine = np.eye(4)
    new_affine[:, 0] = affine[:, sagittal_orientation.normal_component]
    new_affine[:, 1] = affine[:, coronal_orientation.normal_component]
    new_affine[:, 2] = affine[:, axial_orientation.normal_component]
    point = [0, 0, 0, 1]

    # If the orientation of coordinates is inverted, then the origin of the "new" image
    # would correspond to the last voxel of the original image
    # First we need to find which point is the origin point in image coordinates
    # and then transform it in world coordinates
    if not axial_orientation.x_inverted:
        print('inverting axial x')
        new_affine[:, 0] = - new_affine[:, 0]

        # new_affine[0, 3] = - new_affine[0, 3]
    if axial_orientation.y_inverted:
        print('inverting axial y')
        new_affine[:, 1] = - new_affine[:, 1]

        # new_affine[1, 3] = - new_affine[1, 3]
    if coronal_orientation.y_inverted:
        print('inverting axual z')
        new_affine[:, 2] = - new_affine[:, 2]



    print('Asingle')
    print(np.mat(Asingle))
    print('new_affine')
    print(new_affine)


    file_name = file_names[0]
    ds = pydicom.read_file(file_name)
    options = dict()
    options['type'] = 'short'
    options['dimension'] = 3
    #options['space dimension'] = 3
    options['space units'] = ["mm", "mm", "mm"]
    options['space'] = 'right-anterior-superior'
    #options['space'] ='scanner-xyz'
    options['space directions'] = [[1*mzdif, 0, 0],
                                   [0, 1*ds.PixelSpacing[0], 0],
                                   [0, 0, 1*ds.PixelSpacing[1]]]

    # options['space directions'] = [[1*ds.PixelSpacing[0], 0, 0],
    #                                [0, 1*ds.PixelSpacing[1], 0],
    #                                [0, 0, 1*mzdif]]

    # sdirec =                [[1*ds.PixelSpacing[0], 0, 0],
    #                                 [0, 1*ds.PixelSpacing[1], 0],
    #                                 [0, 0, 1*mzdif]]



    matout = matstring(Asingle)

    #options['space directions'] = sdirec #np.matmul(Asingle, sdirec)
    options['measurement frame'] = affine[:3,:3]
    options['kinds'] = ['space', 'space', 'space']
    #options['space origin'] = list(np.array(adj_mat*ds.ImagePositionPatient)[0])
    #options['space origin'] = list(np.array(adj_mat * ds.ImagePositionPatient)[0])
    options['space origin'] = [affine[0,-1], affine[1,-1], affine[2,-1]]
   # options['measurement frame'] = Asingle
    print('affine')
    print(affine)

    #print(options['space origin'])
   # print(options['space directions'])

    #options['space directions'] = np.swapaxes(options['space directions'], 0, 2)

    return options

def matstring(mat):
    matout = ""
    for row in mat:
        row = np.array(row).squeeze()
        rowout = []
        for x in row:
            rowout.append(_convert_to_reproducible_floatingpoint(x))

        matout += f'({",".join(rowout)}) '

    return matout


def _convert_to_reproducible_floatingpoint( x ):
    if type(x) == float:
        value = '{:.16f}'.format(x).rstrip('0').rstrip('.') # Remove trailing zeros, and dot if at end
    else:
        value = str(x)
    return value

def save_nrrd_file(data_3d, filename, options):

    # from matplotlib import pyplot as pl
    # import sys



    #sys.exit()
    # data_3d = np.swapaxes(data_3d, 0, 2)
    #data_3d = np.swapaxes(data_3d, 0, 1)
    # pl.matshow(data_3d[420,:,:].squeeze())
    # pl.matshow(data_3d[:, 420, :].squeeze())
    # pl.matshow(data_3d[:, :, 5].squeeze())
    #
    # pl.show()
    data_3d = data_3d[:, :, :]

    # write the stack files in nrrd format
    nrrd.write(filename, data_3d, options)

# Xform mat:
# 0.997427 0.0691727 -0.0188028 2
# -0.0679978 0.996043 0.0572329 0
# 0.0226873 -0.0558071 0.998184 0
# 0 0 0 1

# def convert_nifti_to_nrrd(nibabel_nifti, dicom_path):
#     data = nib.load(os.path.join(
#         args.nifti_dir,
#         data_idx + '.nii.gz'
#     )).get_data()
#
#     header = nrrd.read_header(os.path.join(
#         args.nrrd_dir,
#         data_idx,
#         'img.nrrd'
#     ))
#     zoom = tuple(
#         (s1 / s2) for s1, s2
#         in zip(header['sizes'], data.shape)
#     )
#     data = ndimage.zoom(
#         data,
#         zoom,
#         order=0,
#         mode='nearest'
#     )
#     assert data.shape == tuple(header['sizes'])
#     nrrd.write(
#         os.path.join(
#             args.output_dir,
#             data_idx + '.nrrd'
#         ),
#         data,
#         header=header
#     )