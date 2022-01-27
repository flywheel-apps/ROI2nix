#!/usr/bin/env python3
import logging
import os

from flywheel_gear_toolkit import GearToolkitContext

import sys

from utils.MeasurementExporter import MeasurementExportFromDicom as DicomExporter
from utils.utils import (
    convert_dicom_to_nifti,
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

        destination_type = "nrrd" if config.get("save_NRRD") else "nifti"

        # The inv_reduced_aff may need to be adjusted for dicoms
        if file_obj["type"] == "nifti":
            log.error("exporting ROI's on nifti files currently not supported")
            sys.exit(1)
        # nii = nib.load(context.get_input_path("Input_File"))
        # TODO: Add check here for save NRRD, curently not possible for only a nifti file due to header stuffies

        elif file_obj["type"] == "dicom":

            # convert dicom-centric data to nifti-centric data
            exporter = DicomExporter(
                fw_client=fw_client,
                input_file_path=context.get_input_path("Input_File"),
                orig_file_type="dicom",
                dest_file_type=destination_type,
                combine=config.get("save_combined_output", False),
                file_object=file_obj,
                work_dir=context.work_dir,
                output_dir=context.output_dir,
            )

        exporter.process_file()

        # #
        # # Calculate the voxel and volume of each ROI by label
        calculate_ROI_volume(exporter.labels, exporter.affine)
        #
        # Output csv file with ROI index, label, num of voxels, and ROI volume
        output_ROI_info(context, exporter.labels)


        # # Write Slicer color table file .cbtl
        if config["save_slicer_color_table"]:
            write_3D_Slicer_CTBL(context, file_input, exporter.labels)

    except Exception as e:
        context.log.exception(e)
        context.log.fatal(
            "Error executing roi2nix.",
        )
        return 1

    context.log.info("roi2nix completed Successfully!")
    return 0


if __name__ == "__main__":
    with GearToolkitContext() as gear_context:
        gear_context.init_logging()
        exit_status = main(gear_context)

    log.info("exit_status is %s", exit_status)
    os.sys.exit(exit_status)
