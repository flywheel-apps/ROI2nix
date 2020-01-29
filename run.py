#!/usr/bin/env python3
import os
import os.path as op
from skimage import draw
import numpy as np
import nibabel as nb
import logging

import flywheel

log = logging.getLogger(__name__)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
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
            shp_idx = [i for i, x in enumerate(orientation_axis) if x == 0]
            orientation_shape = [shape[shp_idx[0]], shape[shp_idx[1]]]

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


if __name__ == '__main__':
    # Activate custom logger
    log_level = logging.INFO
    fmt = '%(asctime)s.%(msecs)03d %(levelname)-8s ' + \
        '[%(name)s %(funcName)s()]: %(message)s'
    logging.basicConfig(level=log_level, format=fmt, datefmt='%H:%M:%S')
    log.info('Log level is {}'.format(log_level))

    # Get the Gear Context
    context = flywheel.GearContext()
    context.log = log
    context.log_config()

    fw = context.client

    # Build, Validate, and execute Parameters Hello World
    try:
        # Get configuration, acquisition, and file info
        file_input = context.get_input('Input_File')
        acquisition = fw.get(file_input['hierarchy']['id'])
        file_obj = acquisition.get_file(file_input['location']['name'])

        nii = nb.load(context.get_input_path('Input_File'))

        labels = []
        if 'label' in context.config.keys():
            labels.append(context.config['label'])
        else:
            # only doing this for toolType=freehand
            # TODO: Consider other closed regions:
            # rectangleRoi, ellipticalRoi
            for roi in file_obj.info['roi']:
                if (roi['toolType'] == 'freehand') and \
                   (roi['label'] not in labels):
                    labels.append(roi['label'])
        data = np.zeros(nii.shape[:3])
        for label in labels:
            idx = 2**labels.index(label)
            log.warning('%s is %i', label, idx)
            data += idx * label2data(label, nii.shape[:3], file_obj.info)
            log.info('max data value is %i.', data.max())

        """
        TODO:
         Here is what I don't know: Does the OHIF viewer display the nifti with
         or without the affine matrix?  If without, then the following is
         appropriate. If with, no... We may need to find another way to
         make sure the coordiates are displayed in the right orientation.
        """
        lbl_nii = nb.Nifti1Pair(data, nii.affine)

        if len(labels) == 1:
            fl_name = labels[0] + '_label_' + file_input['location']['name']
        else:
            fl_name = 'all_labels' + '_' + file_input['location']['name']

        nb.save(lbl_nii, op.join(context.output_dir, fl_name))

        # TODO: If a bunch of labels, do we want a mapping file
        # (e.g. <label> <index> <color>) to translate?

    except Exception as e:
        context.log.fatal(e,)
        context.log.fatal(
            'Error executing roi2nix.',
        )
        os.sys.exit(1)

    context.log.info("roi2nix completed Successfully!")
    os.sys.exit(0)
