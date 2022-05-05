"""
 This module represents functionality used by the `run.py` script for readability
 and encapsulation.  References to `prior art` are given where appropriate.
 Areas for future implementation are noted for ease of prospective implementation.
"""

from collections import OrderedDict
import logging
import numpy as np
import os.path as op
from skimage import draw
import re

rgba_regex = ".*\((?P<R>\d+),\s+?(?P<G>\d+),\s+?(?P<B>\d+),\s+?(?P<A>\d+?\.\d+?)\)"
rgba_regex = re.compile(rgba_regex)

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
    mask = np.zeros(shape, dtype=bool)
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

    mask = np.zeros(shape, dtype=bool)
    mask[fill_row_coords, fill_col_coords] = True

    return mask



def calculate_ROI_volume(labels, affine):
    """
    calculate_ROI_volume calculates the number of voxels and volume in each of
        the ROIs

    Args:
        labels (OrderedDict): The dictionary containing ROI info
        data (numpy.ndarray): The NIfTI data with ROI within
    """
    if len(labels) > 0:
        for _, label_object in labels.items():
            label_object.calc_volume(affine)
            # indx = label_dict["index"]
            # label_data = np.bitwise_and(data, indx)
            # voxels = np.sum(label_data > 0)
            # label_dict["voxels"] = voxels
            # volume = voxels * np.abs(np.linalg.det(affine[:3, :3]))
            # label_dict["volume"] = volume  # mm^3


def output_ROI_info(output_dir, labels):
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
        csv_file = open(op.join(output_dir, "ROI_info.csv"), "w")
        csv_file.writelines(lines)
        csv_file.close()
    else:
        log.warning("There were no labels to process.")


def write_3D_Slicer_CTBL(output_dir, file_input, labels):
    """
    write_3D_Slicer_CTBL saves the label information to a 3D Slicer colortable
        file format.

    Args:
        context (flywheel.gear_context.GearContext): gear context for output_dir
        file_input (flywheel.models.file.File): Input file with metadata
        labels (OrderedDict): Ordered dictionary of ROI label attributes
    """

    output_filename = file_input["location"]["name"]

    if output_filename.endswith(".gz"):
        output_filename = output_filename[: -1 * len(".nii.gz")]
    elif output_filename.endswith(".nii"):
        output_filename = output_filename[: -1 * len(".nii")]
    elif output_filename.endswith(".dicom.zip"):
        output_filename = output_filename[: -1 * len(".dicom.zip")]
    elif output_filename.endswith(".zip"):
        output_filename = output_filename[: -1 * len(".zip")]

    if len(labels) > 0:
        ctbl = open(
            op.join(
                output_dir,
                "ROI_ALL_labels_" + output_filename + ".ctbl",
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
