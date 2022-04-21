import logging
import sys

from utils.MeasurementExporter import MeasurementExport
from utils.roi_tools import (
    calculate_ROI_volume,
    output_ROI_info,
    write_3D_Slicer_CTBL,
)

log = logging.getLogger(__name__)


def main(
    fw_client,
    file_obj,
    save_combined_output,
    save_binary_masks,
    conversion_method,
    input_file_path,
    input_file_object,
    work_dir,
    output_dir,
    destination_type,
    save_slicer_color_table,
):

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
            work_dir=work_dir,
            output_dir=output_dir,
            input_file_path=input_file_path,
            dest_file_type=destination_type,
            combine=save_combined_output,
            bitmask=save_binary_masks,
            method=conversion_method,
        )

    ohifviewer_info, labels, affine = exporter.process_file()

    # #
    # # Calculate the voxel and volume of each ROI by label
    calculate_ROI_volume(labels, affine)
    #
    # Output csv file with ROI index, label, num of voxels, and ROI volume
    output_ROI_info(output_dir, labels)

    # # Write Slicer color table file .cbtl
    if save_slicer_color_table:
        write_3D_Slicer_CTBL(output_dir, input_file_object, labels)
