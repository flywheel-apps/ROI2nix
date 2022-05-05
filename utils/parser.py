from pathlib import Path


def parser(context):
    config = context.config
    fw_client = context.client
    save_combined_output = config.get("save_combined_output", False)
    save_binary_masks = not config.get("save_binary_masks", True)
    conversion_method = config.get("conversion_method")
    input_file_path = Path(context.get_input_path("Input_File"))
    work_dir = Path(context.work_dir)
    output_dir = Path(context.output_dir)
    input_file_object = context.get_input("Input_File")
    destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    save_slicer_color_table = config.get("save_slicer_color_table", False)

    return (
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
