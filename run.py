#!/usr/bin python3
import logging

from flywheel_gear_toolkit import GearToolkitContext

from utils.parser import parser
import utils.ExportRois as ExportRois

log = logging.getLogger(__name__)
print("^^^^ FILE")


def main(
    fw_client,
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

    # Need updated file information.
    file_obj = input_file_object["object"]
    file_obj = fw_client.get_file(file_obj["file_id"])

    # The inv_reduced_aff may need to be adjusted for dicoms
    ExportRois.main(
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
    )


if __name__ == "__main__":

    with GearToolkitContext() as gear_context:
        gear_context.init_logging()
        (
            fw_client,
            save_combined_output,
            save_binary_masks,
            conversion_method,
            input_file_path,
            input_file_object,
            work_dir,
            output_dir,
            destination_type,
            save_slicer_color_table,
        ) = parser(gear_context)

        main(
            fw_client,
            save_combined_output,
            save_binary_masks,
            conversion_method,
            input_file_path,
            input_file_object,
            work_dir,
            output_dir,
            destination_type,
            save_slicer_color_table,
        )
