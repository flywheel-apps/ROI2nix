#!/usr/bin/env python3
import logging
import os

import flywheel
import nibabel as nib
import numpy as np

from utils import (
    calculate_ROI_volume,
    gather_ROI_info,
    label2data,
    output_ROI_info,
    save_bitmasked_ROIs,
    save_single_ROIs,
    write_3D_Slicer_CTBL
)

log = logging.getLogger(__name__)


def main(context):
    fw = context.client
    config = context.config

    try:
        # Get configuration, acquisition, and file info
        file_input = context.get_input('Input_File')
        acquisition = fw.get(file_input['hierarchy']['id'])
        # Need updated file information.
        file_obj = acquisition.get_file(file_input['location']['name'])

        nii = nib.load(context.get_input_path('Input_File'))

        # Collate label, color, and index information into a dictionary keyed
        # by the name of each "label". Enables us to iterate through one "label"
        # at a time.
        labels = gather_ROI_info(file_obj)

        len_labels = len(labels)
        if len_labels > 0:
            context.log.info("Found %s ROI labels", len_labels)
        else:
            context.log.error("Found NO ROI labels")

        # Acquire ROI data
        data = np.zeros(nii.shape[:3], dtype=np.int64)
        for label in labels:
            context.log.info("Getting ROI \"%s\"", label)
            data += labels[label]['index'] * \
                label2data(label, nii.shape[:3], file_obj.info)

        # Output individual ROIs
        save_single_ROIs(
            context,
            file_input,
            labels,
            data,
            nii.affine,
            config['save_binary_masks']
        )

        # Calculate the voxel and volume of each ROI by label
        calculate_ROI_volume(labels, data, nii.affine)

        # Output csv file with ROI index, label, num of voxels, and ROI volume
        output_ROI_info(context, labels)

        # Output all ROIs in one file, if selected
        # TODO: If we want different output styles (last written, 4D Nifti)
        # we would implement that here...with a combo box in the manifest.
        if config['save_combined_output']:
            save_bitmasked_ROIs(context, labels, file_input, data, nii.affine,
                                config['combined_output_size'])

        # Write Slicer color table file .cbtl
        if config['save_slicer_color_table']:
            write_3D_Slicer_CTBL(context, file_input, labels)

    except Exception as e:
        context.log.exception(e)
        context.log.fatal(
            'Error executing roi2nix.',
        )
        return 1

    context.log.info("roi2nix completed Successfully!")
    return 0


if __name__ == '__main__':
    # Activate custom logger
    log_level = logging.INFO
    fmt = '%(asctime)s.%(msecs)03d %(levelname)-8s ' + \
        '[%(name)s %(funcName)s()]: %(message)s'
    logging.basicConfig(level=log_level, format=fmt, datefmt='%H:%M:%S')
    log.info('Log level is {}'.format(log_level))

    with flywheel.GearContext() as gear_context:
        gear_context.log = log
        gear_context.log_config()
        exit_status = main(gear_context)

    log.info('exit_status is %s', exit_status)
    os.sys.exit(exit_status)
