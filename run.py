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


def test_main():

    import os
    import flywheel

    fw = flywheel.Client(os.environ["FWGA_API"])
    # Get configuration, acquisition, and file info
    from pathlib import Path

    parent_acq = "621d449c559d4f2a0d1468e0"
    file_id = "621d449f332f06de9645fd7a"
    file_obj = fw.get_file(file_id)
    input_file = "/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/tests/test_ax_cor_sag/Scans/T1_SE_AX.zip"
    # destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    work_dir = "/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/tests/test_ax_cor_sag/work"
    output_dir = "/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/tests/test_ax_cor_sag/output"
    input_file = Path(input_file)
    work_dir = Path(work_dir)
    output_dir = Path(output_dir)
    destination_type = "nifti"

    combine = False
    bitmask = False
    method = "slicer-dcmtk"

    exporter = MeasurementExport(
        fw_client=fw,
        fw_file=file_obj,
        work_dir=work_dir,
        output_dir=output_dir,
        input_file_path=input_file,
        dest_file_type=destination_type,
        combine=combine,
        bitmask=bitmask,
        method=method,
    )

    ohifviewer_info, labels, affine = exporter.process_file()


def docker_test_main():

    # docker run --rm -ti --entrypoint=/bin/bash -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/ROI2nix/utils/workers:/flywheel/v0/utils/workers -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/ROI2nix/utils/SlicerScripts:/flywheel/v0/converters/scripts -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/tests/test_new_job/ga_cancerarchive2_covid_t2_haste:/flywheel/v0/scrap -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/tests/test_new_job/output:/flywheel/v0/output flywheel/roi2nix:0.5.0



    # docker run --rm -ti --entrypoint=/bin/bash -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/ROI2nix/run.py:/flywheel/v0/run.py -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/ROI2nix/utils/workers:/flywheel/v0/utils/workers -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/ROI2nix/utils/SlicerScripts:/flywheel/v0/converters/scripts -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/tests/test_ax_cor_sag/Scans:/flywheel/v0/scrap -v /Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/tests/test_ax_cor_sag/output:/flywheel/v0/output flywheel/roi2nix:0.5.0

    import os
    import flywheel

    fw = flywheel.Client("ga.ce.flywheel.io:9GmTBZyuzKvizjqz4Y")
    # Get configuration, acquisition, and file info
    from pathlib import Path

    parent_acq = "621d449c559d4f2a0d1468e0"
    file_id = "621d449f332f06de9645fd7a"
    file_obj = fw.get_file(file_id)
    input_file = "/flywheel/v0/scrap/T1_SE_COR.zip"

    # parent_acq = "621d447458481ac69c6437c5"
    # file_id = "621d4476dfdba575dc1468e2"
    # file_obj = fw.get_file(file_id)
    # input_file = "/flywheel/v0/scrap/T1_SE_AX.zip"

    # parent_acq = "621d444658481ac69c6437c4"
    # file_id = "621d4449559d4f2a0d1468df"
    # file_obj = fw.get_file(file_id)
    # input_file = "/flywheel/v0/scrap/T1_SE_SAG.zip"


    # destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    work_dir = "/flywheel/v0/work"
    output_dir = "/flywheel/v0/output"

    input_file = Path(input_file)
    work_dir = Path(work_dir)
    output_dir = Path(output_dir)
    destination_type = "nifti"

    combine = False
    bitmask = False
    method = "dicom2nifti"

    exporter = MeasurementExport(
        fw_client=fw,
        fw_file=file_obj,
        work_dir=work_dir,
        output_dir=output_dir,
        input_file_path=input_file,
        dest_file_type=destination_type,
        combine=combine,
        bitmask=bitmask,
        method=method,
    )

    ohifviewer_info, labels, affine = exporter.process_file()


if __name__ == "__main__":
    # docker_test_main()
    #test_main()

    with GearToolkitContext() as gear_context:
        gear_context.init_logging()
        exit_status = main(gear_context)

    log.info("exit_status is %s", exit_status)
    os.sys.exit(exit_status)
