import os
from tests import setup_data as sd

os.environ["SCRIPT_DIR"] = os.path.join(sd.cwd, "utils/SlicerScripts")

import glob
import logging
from pathlib import Path
import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from utils.workers import Converters
from utils.objects.Conversion import NRRD_TYPE, NIFTI_TYPE

SLICER_SCRIPT = Converters.SLICER_SCRIPT


log = logging.getLogger(__name__)


def test_BaseCollector_factory():

    tests = [
        ("dcm2niix", Converters.dcm2niix),
        ("slicer-dcmtk", Converters.slicer_dcmtk),
        ("slicer-gdcm", Converters.slicer_gdcm),
        ("slicer-arch", Converters.slicer_arch),
        ("plastimatch", Converters.plastimatch),
        ("dicom2nifti", Converters.dicom2nifti),
    ]

    conversion = MagicMock()

    for test in tests:
        test_type, test_class = test
        converter = Converters.BaseConverter.factory(
            type_=test_type,
            orig_dir=None,
            roi_dir=None,
            output_dir=None,
            conversion=conversion,
        )
        assert isinstance(converter, test_class)

    test_type = "random_type"
    with pytest.raises(NotImplementedError):
        Converters.BaseConverter.factory(
            type_=test_type,
            orig_dir=None,
            roi_dir=None,
            output_dir=None,
            conversion=conversion,
        )


def test_dcm2niix_run_command():

    type = "dcm2niix"
    conversion = MagicMock()
    converter = Converters.BaseConverter.factory(
        type_=type, orig_dir=None, roi_dir=None, output_dir=None, conversion=conversion
    )

    with patch("utils.workers.Converters.sp", MagicMock()):
        converter.run_command("test")
        Converters.sp.Popen.assert_called_with("test")


def test_dcm2niix_make_command_NRRD():

    type = "dcm2niix"
    conversion = MagicMock()
    conversion.ext = NRRD_TYPE
    output_filename = os.path.join(sd.WORKING_DIR, "output_filename")

    orig_dir = Path("orig_dir")
    roi_dir = Path("roi_dir")
    output_dir = Path("output_dir")

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    command = converter.make_command(output_filename)
    assert command == [
        "dcm2niix",
        "-o",
        output_dir.as_posix(),
        "-f",
        output_filename,
        "-b",
        "n",
        "-e",
        "y",
        roi_dir.as_posix(),
    ]


def test_dcm2niix_make_command_NIFTI():
    type = "dcm2niix"
    conversion = MagicMock()
    conversion.ext = "nifti"
    output_filename = os.path.join(sd.WORKING_DIR, "output_filename")

    orig_dir = Path("orig_dir")
    roi_dir = Path("roi_dir")
    output_dir = Path("output_dir")

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    command = converter.make_command(output_filename)
    assert command == [
        "dcm2niix",
        "-o",
        output_dir.as_posix(),
        "-f",
        output_filename,
        "-b",
        "n",
        roi_dir.as_posix(),
    ]


def test_slicer_dcmtk_make_command():
    type = "slicer-dcmtk"
    conversion = MagicMock()
    conversion.ext = "ext"
    output_filename = os.path.join(sd.WORKING_DIR, "output_filename")

    orig_dir = Path("orig_dir")
    roi_dir = Path("roi_dir")
    output_dir = Path("output_dir")

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    command = converter.make_command(output_filename)
    assert command == [
        "xvfb-run",
        "Slicer",
        "--python-script",
        Converters.SLICER_SCRIPT,
        "--dcmtk",
        "--input",
        roi_dir.as_posix(),
        "--output",
        output_dir.as_posix(),
        "--filename",
        output_filename + "." + conversion.ext,
    ]


def test_slicer_gdcm_make_command():
    type = "slicer-gdcm"
    conversion = MagicMock()
    conversion.ext = "ext"
    output_filename = os.path.join(sd.WORKING_DIR, "output_filename")

    orig_dir = Path("orig_dir")
    roi_dir = Path("roi_dir")
    output_dir = Path("output_dir")

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    command = converter.make_command(output_filename)
    assert command == [
        "xvfb-run",
        "Slicer",
        "--python-script",
        Converters.SLICER_SCRIPT,
        "--gdcm",
        "--input",
        roi_dir.as_posix(),
        "--output",
        output_dir.as_posix(),
        "--filename",
        output_filename + "." + conversion.ext,
    ]


def test_slicer_arch_make_command():
    type = "slicer-arch"
    conversion = MagicMock()
    conversion.ext = "ext"
    output_filename = os.path.join(sd.WORKING_DIR, "output_filename")

    orig_dir = Path("orig_dir")
    roi_dir = Path("roi_dir")
    output_dir = Path("output_dir")

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    command = converter.make_command(output_filename)
    assert command == [
        "xvfb-run",
        "Slicer",
        "--python-script",
        Converters.SLICER_SCRIPT,
        "--archetype",
        "--input",
        roi_dir.as_posix(),
        "--output",
        output_dir.as_posix(),
        "--filename",
        output_filename + "." + conversion.ext,
    ]


def test_plastimatch_make_command():
    type = "plastimatch"
    conversion = MagicMock()
    conversion.ext = "ext"
    output_filename = os.path.join(sd.WORKING_DIR, "output_filename")

    orig_dir = Path("orig_dir")
    roi_dir = Path("roi_dir")
    output_dir = Path("output_dir")

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    command = converter.make_command(output_filename)

    assert command == [
        "plastimatch",
        "convert",
        "--input",
        roi_dir.as_posix(),
        "--output-img",
        os.path.join(output_dir.as_posix(), output_filename + f".{converter.ext}"),
    ]


def test_dicom2nifti_make_command():
    type = "dicom2nifti"
    conversion = MagicMock()
    conversion.ext = "ext"
    output_filename = os.path.join(sd.WORKING_DIR, "output_filename")

    orig_dir = Path("orig_dir")
    roi_dir = Path("roi_dir")
    output_dir = Path("output_dir")

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    command = converter.make_command()

    assert command == ["dicom2nifti", roi_dir.as_posix(), output_dir.as_posix()]


def test_remove_accents():
    test_string = "Ťéṡṯ\hyphen-string/NO hyphen"
    cleaned_string = Converters._remove_accents(test_string)
    assert cleaned_string == "testhyphen-stringno_hyphen"


def test_dicom2nifti_guess_dicom2nifti_outputname():
    dicom_dir = sd.unzip_t1()

    type = "dicom2nifti"
    conversion = MagicMock()
    conversion.ext = "ext"
    output_filename = os.path.join(sd.WORKING_DIR, "output_filename")

    orig_dir = Path("orig_dir")
    roi_dir = Path(dicom_dir)
    output_dir = Path("output_dir")

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    base_filename = converter.guess_dicom2nifti_outputname()
    assert base_filename == "2_t2_tse_coronal_512"
    sd.clean_working_dir()


def test_dicom2nifti_rename_dicom2nifti_output():
    dicom_dir = sd.unzip_t1()
    nifti_file = sd.setup_nifti("2_t2_tse_coronal_512.nii.gz", unzip=True)

    type = "dicom2nifti"
    conversion = MagicMock()
    conversion.ext = NIFTI_TYPE

    orig_dir = Path("orig_dir")
    roi_dir = Path(dicom_dir)
    output_dir = Path(sd.WORKING_DIR)

    converter = Converters.BaseConverter.factory(
        type_=type,
        orig_dir=orig_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        conversion=conversion,
    )

    converter.rename_dicom2nifti_output("test_output")

    old_file = os.path.join(sd.WORKING_DIR, "2_t2_tse_coronal_512.nii")
    new_file = os.path.join(sd.WORKING_DIR, "test_output.nii")
    assert not os.path.exists(old_file)
    assert os.path.exists(new_file)

    sd.clean_working_dir()
