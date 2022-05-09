import sys
sys.path.append('/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/roi2nix/ROI2nix')
sys.path.append('/flywheel/v0')

from utils.MeasurementExporter import MeasurementExport

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
    method = "dcm2niix"

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

test_main()