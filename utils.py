"""
 This module represents functionality used by the `run.py` script for readability
 and encapsulation.  References to `prior art` are given where appropriate.
 Areas for future implementation are noted for ease of prospective implementation.
"""

import logging
import os.path as op
import re
from collections import OrderedDict

import nibabel as nib
import numpy as np
from skimage import draw

log = logging.getLogger(__name__)


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


def label2data(label, shape, info, inv_reduced_aff):
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
                        fill_roi_slice(
                            data,
                            roi["imagePath"],
                            roi["handles"],
                            inv_reduced_aff,
                            roi_type=roi_type,
                        )
    # deprioritize any OHIF Meteor annotations...
    # ROI2Nix will not do both
    # TODO: Deprecate OHIF Meteor functionality
    elif "roi" in info:
        for roi in info["roi"]:
            if roi["label"] == label:
                fill_roi_slice(
                    data,
                    roi["imagePath"],
                    roi["handles"],
                    inv_reduced_aff,
                    reactOHIF=False,
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


def save_single_ROIs(context, file_input, labels, data, affine, binary):
    """
    Output_Single_ROIs saves single ROIs to their own file. If `binary` is
        true, we have a binary mask (0,1) for the file, else the value is
        the bitmasked (power-of-two).

    Args:
        context (flywheel.gear_context.GearContext): Gear Context
        labels (OrderedDict): Label data
        data (numpy.ndarray): NIfTI data object
        affine (numpy.ndarray): NIfTI affine
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

            label_nii = nib.Nifti1Pair(export_data, affine)
            label_out = re.sub("[^0-9a-zA-Z]+", "_", label)
            filename = "ROI_" + label_out + "_" + file_input["location"]["name"]
            # If the original file_input was an uncompressed NIfTI
            # ensure compression
            if filename[-3:] == "nii":
                filename += ".gz"

            nib.save(label_nii, op.join(context.output_dir, filename))
    else:
        log.warning("No ROIs were found for this image.")


def save_bitmasked_ROIs(
    context, labels, file_input, data, affine, combined_output_size
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
        all_labels_nii = nib.Nifti1Pair(data.astype(np_type), affine)
        filename = "ROI_ALL_" + file_input["location"]["name"]

        # If the original file_input was an uncompressed NIfTI
        # ensure compression
        if filename[-3:] == "nii":
            filename += ".gz"

        nib.save(all_labels_nii, op.join(context.output_dir, filename))
    else:
        log.warning("There are not enough ROIs to save an aggregate.")


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
