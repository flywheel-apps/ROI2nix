import os.path as op
from skimage import draw
import numpy as np
import nibabel as nib
import logging
from collections import OrderedDict

log = logging.getLogger(__name__)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """
    poly2mask takes polygon vertex coordinates and turns them into a filled
        polygon

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
        vertex_row_coords,
        vertex_col_coords,
        shape
    )
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def label2data(label, shape, info):
    """
    label2data gives the roi data block for a nifti file with `shape` and
    `info`

    This iterates through polygons drawn in any orientation and renders the
    shape contained in the indicated slice of a numpy array. This routine does
    not yet do ellipses or rectangles.

    Args:
        label (string): The label to convert the polygon data into nifti data
        shape (list): The shape of the nifti data (e.g. [256,256,256])
        info (dict): The `info` object of the flywheel file object

    Returns:
        numpy.array: A numpy array of `shape` with values `True` where `label`
            has been segmented and `False` on the outside.

    """
    data = np.zeros(shape, dtype=np.bool)
    for roi in info['roi']:
        if roi["label"] == label:
            # Find orientation [Axial, Sagital, Coronal]
            img_path = roi["imagePath"]
            # orientation character gives us a direction perpendicular
            orientation_char = img_path[img_path.find('#') + 1]

            orientation_coordinate = int(img_path[
                img_path.find('#') + 3:
                img_path.find(',')
            ])

            if orientation_char == 'z':
                orientation_axis = [0, 0, 1]
                orientation_slice = data[:, :, orientation_coordinate]
            elif orientation_char == 'y':
                orientation_axis = [0, 1, 0]
                orientation_slice = data[:, orientation_coordinate, :]
            elif orientation_char == 'x':
                orientation_axis = [1, 0, 0]
                orientation_slice = data[orientation_coordinate, :, :]
            else:
                log.warning('Orientation character not recognized.')
                orientation_axis = ''
                orientation_slice = [0, 0, 0]

            # initialize x,y-coordinate lists
            shp_indx = [i for i, x in enumerate(orientation_axis) if x == 0]
            orientation_shape = [shape[shp_indx[0]], shape[shp_indx[1]]]

            # Initialize x,y coordinates for each polygonal point
            X = []
            Y = []
            if isinstance(roi["handles"], list):
                for h in roi['handles']:
                    if orientation_char == 'x':
                        X.append(
                            orientation_shape[0] - h['x']
                        )
                    else:
                        X.append(h['x'])
                    Y.append(
                        orientation_shape[1] - h['y']
                    )
            X.append(X[0])
            Y.append(Y[0])

            # If this slice already has data, we need to have the logical or
            # of that data and the new data
            orientation_slice[:, :] = np.logical_or(
                poly2mask(X, Y, orientation_shape),
                orientation_slice[:, :]
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

    # only doing this for toolType=freehand
    # TODO: Consider other closed regions:
    # rectangleRoi, ellipticalRoi
    if 'roi' in file_obj.info.keys():
        for roi in file_obj.info['roi']:
            if (roi['toolType'] == 'freehand') and \
                    (roi['label'] not in labels.keys()):
                # Only if annotation type is a polygon, then grab the
                # label, create a 2^x index for bitmasking, grab the color
                # hash (e.g. #fbbc05), and translate it into RGB
                labels[roi['label']] = {
                    'index': int(2**(len(labels))),
                    'color': roi['color'],
                    'RGB': [
                        int(roi['color'][i: i + 2], 16)
                        for i in [1, 3, 5]
                    ]
                }
    else:
        log.warning('No ROIs were found for this image.')

    return labels


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
            indx = labels[label]['index']
            if binary:
                modifier = indx
            else:
                modifier = 1
            export_data = np.bitwise_and(data, indx).astype(np.int8) / modifier
            voxels = np.sum(export_data > 0)
            labels[label]['voxels'] = voxels
            volume = voxels * np.linalg.det(affine[:3, :3])
            labels[label]['volume'] = volume  # mm^3
            label_nii = nib.Nifti1Pair(
                export_data,
                affine
            )
            nib.save(
                label_nii,
                op.join(
                    context.output_dir,
                    'ROI_' + label + '_' + file_input['location']['name']
                )
            )
    else:
        log.warning('No ROIs were found for this image.')


def save_bitmasked_ROIs(context, labels, file_input, data, affine):
    """
    save_bitmasked_ROIs saves all ROIs rendered into a bitmasked NIfTI file.

    Args:
        context (flywheel.gear_context.GearContext): Gear context
        labels (OrderedDict): The label attributes.
        file_input (flywheel.models.file.File): Input File object
        data (numpy.ndarray): The nifti data object
        affine (numpy.ndarray): The nifti affine
    """

    if len(labels) > 1:
        all_labels_nii = nib.Nifti1Pair(data.astype(np.int8), affine)
        fl_name = 'ROI_ALL_' + file_input['location']['name']
        nib.save(all_labels_nii, op.join(context.output_dir, fl_name))
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
            lines.append(
                '{},{},{},{}\n'.format(label, index, voxels, volume)
            )
        csv_file = open(op.join(context.output_dir, 'ROI_info.csv'), 'w')
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
                'ROI_ALL_labels_' +
                file_input['location']['name'][:-7] +
                '.ctbl'
            ),
            'w'
        )
        for label in labels:
            ctbl.write(
                '{} '.format(labels[label]["index"]) +
                '{} '.format(label) +
                '{} '.format(labels[label]["RGB"][0]) +
                '{} '.format(labels[label]["RGB"][1]) +
                '{} '.format(labels[label]["RGB"][2]) +
                '255\n'
            )

        ctbl.close()