from utils.MeasurementExporter import MeasurementExport
import os
import flywheel
from pathlib import Path
from tests import setup_data as sd
import glob


def test_dcm2niix_nifti():

    sd.clean_working_dir()

    input_file = sd.get_zipped_t1()
    work_dir = Path(sd.WORKING_DIR) / "work"
    work_dir.mkdir(exist_ok=True)
    output_dir = Path(sd.WORKING_DIR) / "output"
    output_dir.mkdir(exist_ok=True)

    fw = flywheel.Client()
    # Get configuration, acquisition, and file info
    file_id = "626037fe7cc96261fa295c75"
    file_obj = fw.get_file(file_id)

    # destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    input_file = Path(input_file)
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
    globpath = output_dir / "*"
    output_files = glob.glob(globpath.as_posix())
    assert len(output_files) == 3

    output_files = [os.path.basename(a) for a in output_files]
    doggy = [d for d in output_files if d.startswith("ROI_Doggy")]
    potato = [d for d in output_files if d.startswith("ROI_Potato")]
    lesion = [d for d in output_files if d.startswith("ROI_Lesion")]

    assert len(doggy) == 1
    assert len(potato) == 1
    assert len(lesion) == 1

    sd.clean_working_dir()


def test_slicer_dcmtk_nifti():

    sd.clean_working_dir()

    input_file = sd.get_zipped_t1()
    work_dir = Path(sd.WORKING_DIR) / "work"
    work_dir.mkdir(exist_ok=True)
    output_dir = Path(sd.WORKING_DIR) / "output"
    output_dir.mkdir(exist_ok=True)

    fw = flywheel.Client()
    # Get configuration, acquisition, and file info
    file_id = "626037fe7cc96261fa295c75"
    file_obj = fw.get_file(file_id)

    # destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    input_file = Path(input_file)
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
    globpath = output_dir / "*"
    output_files = glob.glob(globpath.as_posix())
    assert len(output_files) == 3

    output_files = [os.path.basename(a) for a in output_files]
    doggy = [d for d in output_files if d.startswith("ROI_Doggy")]
    potato = [d for d in output_files if d.startswith("ROI_Potato")]
    lesion = [d for d in output_files if d.startswith("ROI_Lesion")]

    assert len(doggy) == 1
    assert len(potato) == 1
    assert len(lesion) == 1

    sd.clean_working_dir()


def test_slicer_gdcm_nifti():

    sd.clean_working_dir()

    input_file = sd.get_zipped_t1()
    work_dir = Path(sd.WORKING_DIR) / "work"
    work_dir.mkdir(exist_ok=True)
    output_dir = Path(sd.WORKING_DIR) / "output"
    output_dir.mkdir(exist_ok=True)

    fw = flywheel.Client()
    # Get configuration, acquisition, and file info
    file_id = "626037fe7cc96261fa295c75"
    file_obj = fw.get_file(file_id)

    # destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    input_file = Path(input_file)
    destination_type = "nifti"
    combine = False
    bitmask = False
    method = "slicer-gdcm"

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
    globpath = output_dir / "*"
    output_files = glob.glob(globpath.as_posix())
    assert len(output_files) == 3

    output_files = [os.path.basename(a) for a in output_files]
    doggy = [d for d in output_files if d.startswith("ROI_Doggy")]
    potato = [d for d in output_files if d.startswith("ROI_Potato")]
    lesion = [d for d in output_files if d.startswith("ROI_Lesion")]

    assert len(doggy) == 1
    assert len(potato) == 1
    assert len(lesion) == 1

    sd.clean_working_dir()


def test_slicer_arch_nifti():

    sd.clean_working_dir()

    input_file = sd.get_zipped_t1()
    work_dir = Path(sd.WORKING_DIR) / "work"
    work_dir.mkdir(exist_ok=True)
    output_dir = Path(sd.WORKING_DIR) / "output"
    output_dir.mkdir(exist_ok=True)

    fw = flywheel.Client()
    # Get configuration, acquisition, and file info
    file_id = "626037fe7cc96261fa295c75"
    file_obj = fw.get_file(file_id)

    # destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    input_file = Path(input_file)
    destination_type = "nifti"
    combine = False
    bitmask = False
    method = "slicer-arch"

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
    globpath = output_dir / "*"
    output_files = glob.glob(globpath.as_posix())
    assert len(output_files) == 3

    output_files = [os.path.basename(a) for a in output_files]
    doggy = [d for d in output_files if d.startswith("ROI_Doggy")]
    potato = [d for d in output_files if d.startswith("ROI_Potato")]
    lesion = [d for d in output_files if d.startswith("ROI_Lesion")]

    assert len(doggy) == 1
    assert len(potato) == 1
    assert len(lesion) == 1

    sd.clean_working_dir()


def test_plastimatch_nifti():

    sd.clean_working_dir()

    input_file = sd.get_zipped_t1()
    work_dir = Path(sd.WORKING_DIR) / "work"
    work_dir.mkdir(exist_ok=True)
    output_dir = Path(sd.WORKING_DIR) / "output"
    output_dir.mkdir(exist_ok=True)

    fw = flywheel.Client()
    # Get configuration, acquisition, and file info
    file_id = "626037fe7cc96261fa295c75"
    file_obj = fw.get_file(file_id)

    # destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    input_file = Path(input_file)
    destination_type = "nifti"
    combine = False
    bitmask = False
    method = "plastimatch"

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
    globpath = output_dir / "*"
    output_files = glob.glob(globpath.as_posix())
    assert len(output_files) == 3

    output_files = [os.path.basename(a) for a in output_files]
    doggy = [d for d in output_files if d.startswith("ROI_Doggy")]
    potato = [d for d in output_files if d.startswith("ROI_Potato")]
    lesion = [d for d in output_files if d.startswith("ROI_Lesion")]

    assert len(doggy) == 1
    assert len(potato) == 1
    assert len(lesion) == 1

    sd.clean_working_dir()


def test_dicom2nifti_nifti():

    sd.clean_working_dir()

    input_file = sd.get_zipped_t1()
    work_dir = Path(sd.WORKING_DIR) / "work"
    work_dir.mkdir(exist_ok=True)
    output_dir = Path(sd.WORKING_DIR) / "output"
    output_dir.mkdir(exist_ok=True)

    fw = flywheel.Client()
    # Get configuration, acquisition, and file info
    file_id = "626037fe7cc96261fa295c75"
    file_obj = fw.get_file(file_id)

    # destination_type = "nrrd" if config.get("save_NRRD") else "nifti"
    input_file = Path(input_file)
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
    globpath = output_dir / "*"
    output_files = glob.glob(globpath.as_posix())
    assert len(output_files) == 3

    output_files = [os.path.basename(a) for a in output_files]
    doggy = [d for d in output_files if d.startswith("ROI_Doggy")]
    potato = [d for d in output_files if d.startswith("ROI_Potato")]
    lesion = [d for d in output_files if d.startswith("ROI_Lesion")]

    assert len(doggy) == 1
    assert len(potato) == 1
    assert len(lesion) == 1

    sd.clean_working_dir()
