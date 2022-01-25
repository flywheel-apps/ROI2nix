#!/usr/bin/env python3
import logging
import os

import nibabel as nib
import numpy as np
from flywheel_gear_toolkit import GearToolkitContext
import glob
import pydicom
from pathlib import Path

from utils import (
    calculate_ROI_volume,
    convert_dicom_to_nifti,
    gather_ROI_info,
    label2data,
    output_ROI_info,
    write_3D_Slicer_CTBL,
    calculate_transformation_matrix,
    save_rois,
    save_dicom_out
)

log = logging.getLogger(__name__)


def main(context):
    config = context.config

    try:
        # Get configuration, acquisition, and file info
        file_input = context.get_input("Input_File")
        # Need updated file information.
        file_obj = file_input["object"]

        # The inv_reduced_aff may need to be adjusted for dicoms
        adjustment_matrix = np.eye(3)
        if file_obj["type"] == "nifti":
            nii = nib.load(context.get_input_path("Input_File"))
            # TODO: Add check here for save NRRD, curently not possible for only a nifti file due to header stuffies


        elif file_obj["type"] == "dicom":
            # convert dicom-centric data to nifti-centric data
            dicom_output, dicom_dir, file_obj = convert_dicom_to_nifti(
                context, input_name="Input_File"
            )
        #     # If the DICOM is Axial or Coronal (perp_char is "z" or "y") modify the
        #     # below adjustment matrix to indicate the left/right origin of the x-axis.
        #     # otherwise, do nothing.
        #     if perp_char in ["z", "y"]:
        #         adjustment_matrix[:, 0] = -1 * adjustment_matrix[:, 0]
        #
        # inv_reduced_aff = calculate_transformation_matrix(adjustment_matrix, nii.affine)

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
        globdir = dicom_dir / '*.dcm'
        dicom_files = glob.glob(globdir.as_posix())
        print(globdir)
        print(dicom_files)
        dicom_files.sort()
        dicom_files = [Path(d) for d in dicom_files]
        dicoms = [pydicom.read_file(d) for d in dicom_files]
        dicom_sops = [d.SOPInstanceUID for d in dicoms]

        example_dicom = dicoms[0]
        shape = [example_dicom.pixel_array.shape[0], example_dicom.pixel_array.shape[1], len(dicom_files)]

        data = np.zeros(shape, dtype=np.int64)
        #data = np.zeros(nii.shape[:3], dtype=np.int64)

        for label in labels:
            context.log.info('Getting ROI "%s"', label)
            data += labels[label]["index"] * label2data(
                label, file_obj["info"], dicoms, data.shape
            )
        save_dicom_out(dicom_files, dicoms, dicom_output, data)
        #save_rois(context.output_dir, file_input, labels, data, dicom_files, dicom_output, config)

        #
        # # Calculate the voxel and volume of each ROI by label
        # calculate_ROI_volume(labels, data, nii.affine)
        #
        # # Output csv file with ROI index, label, num of voxels, and ROI volume
        # output_ROI_info(context, labels)
        #
        #
        #
        # # Write Slicer color table file .cbtl
        # if config["save_slicer_color_table"]:
        #     write_3D_Slicer_CTBL(context, file_input, labels)

    except Exception as e:
        context.log.exception(e)
        context.log.fatal("Error executing roi2nix.",)
        return 1

    context.log.info("roi2nix completed Successfully!")
    return 0


if __name__ == "__main__":
    with GearToolkitContext() as gear_context:
        gear_context.init_logging()
        exit_status = main(gear_context)

    log.info("exit_status is %s", exit_status)
    os.sys.exit(exit_status)
