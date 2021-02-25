#!/usr/bin/env python3
import logging
import os

import nibabel as nib
import numpy as np
from flywheel_gear_toolkit import GearToolkitContext

from utils import (
    calculate_ROI_volume,
    convert_dicom_to_nifti,
    gather_ROI_info,
    label2data,
    output_ROI_info,
    save_bitmasked_ROIs,
    save_single_ROIs,
    write_3D_Slicer_CTBL,
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

        elif file_obj["type"] == "dicom":
            # convert dicom-centric data to nifti-centric data
            nii, file_obj, perp_char = convert_dicom_to_nifti(
                context, input_name="Input_File"
            )
            # If the DICOM is Axial or Coronal (perp_char is "z" or "y") modify the
            # below adjustment matrix to indicate the left/right origin of the x-axis.
            # otherwise, do nothing.
            if perp_char in ["z", "y"]:
                adjustment_matrix[:, 0] = -1 * adjustment_matrix[:, 0]

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
                        nii.affine[:3, :3],
                        # Generate the [1/norm(),...] for each column
                        # Take the norm of the column vectors.. this is the pixel width
                        np.diag(1.0 / np.linalg.norm(nii.affine[:3, :3], axis=0)),
                    )
                )
            ),
            adjustment_matrix,
        )

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
            context.log.info('Getting ROI "%s"', label)
            data += labels[label]["index"] * label2data(
                label, nii.shape[:3], file_obj["info"], inv_reduced_aff
            )

        # Output individual ROIs
        save_single_ROIs(
            context, file_input, labels, data, nii.affine, config["save_binary_masks"]
        )

        # Calculate the voxel and volume of each ROI by label
        calculate_ROI_volume(labels, data, nii.affine)

        # Output csv file with ROI index, label, num of voxels, and ROI volume
        output_ROI_info(context, labels)

        # Output all ROIs in one file, if selected
        # TODO: If we want different output styles (last written, 4D Nifti)
        # we would implement that here...with a combo box in the manifest.
        if config["save_combined_output"]:
            save_bitmasked_ROIs(
                context,
                labels,
                file_input,
                data,
                nii.affine,
                config["combined_output_size"],
            )

        # Write Slicer color table file .cbtl
        if config["save_slicer_color_table"]:
            write_3D_Slicer_CTBL(context, file_input, labels)

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
