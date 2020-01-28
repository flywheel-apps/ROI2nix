#!/usr/bin/env python3
import os, os.path as op
import json
import subprocess as sp
import copy
import shutil
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
    label2data gives the roi data block for a nifti file with `shape` and `info`

    Args:
        label (string): The label to convert the polygon data into nifti data
        shape (list): The shape of the nifti data (e.g. [256,256,256])
        info (dict): The `info` object of the flywheel file object
    """
    data = np.zeros(shape)
    for roi in info['roi']:
        if roi["label"] == label:
            # Find orientation [Axial, Sagital, Coronal]
            img_path = roi["imagePath"]
            # orientation character gives us a direction perpendicular
            orientation_char = img_path[img_path.find('#')+1]

            orientation_coordinate = int(img_path[
                img_path.find('#')+3:
                img_path.find(',')
            ])

            if orientation_char == 'z':
                orientation = 'Axial'
                orientation_axis = [0,0,1]
                orientation_slice = data[:, :, orientation_coordinate]
            elif orientation_char == 'y':
                orientation = 'Sagitall'
                orientation_axis = [0,1,0]
                orientation_slice = data[:, orientation_coordinate, :]
            elif orientation_char == 'x':
                orientation_axis = [1,0,0]
                orientation = 'Coronal'
                orientation_slice = data[orientation_coordinate, :, :]
            else:
                log.warning('Orientation character not recognized.')
                orientation = ''
                orientation_axis = ''
                orientation_slice = [0, 0, 0]
            
            # initialize x,y-coordinate lists
            X = []
            Y = []
            if type(roi["handles"]) == list:
                for h in roi['handles']:
                    X.append(h['x'])
                    Y.append(h['y'])
            X.append(X[0])
            Y.append(Y[0])

            shp_idx = [i for i,x in enumerate(orientation_axis) if x==0]
            orientation_shape = [shape[shp_idx[0]], shape[shp_idx[1]]]
            orientation_slice[:,:] = poly2mask(X, Y, orientation_shape)
    
    return data


if __name__ == '__main__':
    # Get the Gear Context
    context = flywheel.GearContext()
    fw = context.client

    # Activate custom logger
    log_name = '[roi2nix]'
    log_level = logging.INFO
    fmt = '%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s %(funcName)s()]: %(message)s'
    logging.basicConfig(level=log_level, format=fmt, datefmt='%H:%M:%S')
    context.log = logging.getLogger(log_name)
    context.log.critical('{} log level is {}'.format(log_name, log_level))

    context.log_config()

    # Build, Validate, and execute Parameters Hello World 
    try:
        # Get configuration, acquisition, and file info
        file_input = context.get_input('Input_File')
        acquisition = fw.get(file_input['hierarchy']['id'])
        file_obj = acquisition.get_file(file_input['location']['name'])

        nii = nb.load(context.get_input_path('Input_File'))

        labels = []
        if context.get_context_value('label'):
            labels.append(context.get_context_value('label'))
        else:
            for roi in file_obj.info['roi']:
                if roi['label'] not in labels:
                    labels.append(roi['label'])
        data = np.zeros(nii.shape[:3])
        for label in labels:
            idx = 2**labels.index(label)
            data += idx * label2data(label, nii.shape[:3], file_obj.info)
        
        lbl_nii = nb.Nifti1Pair(data,nii.affine)

        if len(labels) ==  1:
            fl_name = labels[0] + '_' + file_input['location']['name']
        else:
            fl_name = 'all_labels' + '_' + file_input['location']['name']

        nb.save(lbl_nii, fl_name)

        acquisition.upload_file(fl_name)


        
    except Exception as e:
        context.log.fatal(e,)
        context.log.fatal(
            'Error executing roi2nix.',
        )
        os.sys.exit(1)

    context.log.info("roi2nix completed Successfully!")
    os.sys.exit(0)