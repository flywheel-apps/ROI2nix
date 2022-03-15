#!/usr/bin/env python3
import logging
import os
from pathlib import Path
import sys

from flywheel_gear_toolkit import GearToolkitContext

from utils.MeasurementExporter import MeasurementExport
from utils.roi_tools import (
    calculate_ROI_volume,
    output_ROI_info,
    write_3D_Slicer_CTBL,
)

log = logging.getLogger(__name__)


def main(context):
    config = context.config
    fw_client = context.client

    try:
        # Get configuration, acquisition, and file info
        file_input = context.get_input("Input_File")
        # Need updated file information.
        file_obj = file_input["object"]
        file_obj = fw_client.get_file(file_obj["file_id"])

        destination_type = "nrrd" if config.get("save_NRRD") else "nifti"

        # The inv_reduced_aff may need to be adjusted for dicoms
        if file_obj["type"] == "nifti":
            log.error("exporting ROI's on nifti files currently not supported")
            sys.exit(1)
        # nii = nib.load(context.get_input_path("Input_File"))
        # TODO: Add check here for save NRRD, curently not possible for only a nifti file due to header stuffies

        elif file_obj["type"] == "dicom":

            # convert dicom-centric data to nifti-centric data
            exporter = MeasurementExport(
                fw_client=fw_client,
                fw_file=file_obj,
                work_dir=Path(context.work_dir),
                output_dir=Path(context.output_dir),
                input_file_path=Path(context.get_input_path("Input_File")),
                dest_file_type=destination_type,
                combine=config.get("save_combined_output", False),
                bitmask=not config.get("save_binary_masks", True),
                method=config.get("conversion_method"),
            )

        ohifviewer_info, labels, affine = exporter.process_file()

        # #
        # # Calculate the voxel and volume of each ROI by label
        calculate_ROI_volume(labels, affine)
        #
        # Output csv file with ROI index, label, num of voxels, and ROI volume
        output_ROI_info(context, labels)

        # # Write Slicer color table file .cbtl
        if config["save_slicer_color_table"]:
            write_3D_Slicer_CTBL(context, file_input, labels)

    except Exception as e:
        log.exception(e)
        log.fatal(
            "Error executing roi2nix.",
        )
        return 1

    log.info("roi2nix completed Successfully!")
    return 0



if __name__ == "__main__":

    with GearToolkitContext() as gear_context:
        gear_context.init_logging()
        exit_status = main(gear_context)

    log.info("exit_status is %s", exit_status)
    os.sys.exit(exit_status)
