import glob
import logging
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch
from utils.workers import Preppers
from tests import setup_data as sd

log = logging.getLogger(__name__)


def test_BasePrepper_factory():

    tests = [
        ("dicom", Preppers.PrepDicom),
        ("nifti-notimplemented", Preppers.PrepNifti),
    ]

    for test in tests:
        test_type, test_class = test
        prepper = Preppers.BasePrepper.factory(
            type_=test_type, work_dir="", input_file_path=None
        )
        assert isinstance(prepper, test_class)

    with pytest.raises(NotImplementedError):
        Preppers.BasePrepper.factory(type_="invalid", work_dir="", input_file_path=None)


def test_PrepDicom_move_dicoms_to_workdir_zipped():

    prepper = Preppers.PrepDicom(
        work_dir=sd.WORKING_DIR, input_file_path=sd.DICOM_TEST_DATA
    )
    prepper.move_dicoms_to_workdir()

    expected_dir = prepper.orig_dir
    print(expected_dir)
    assert os.path.exists(expected_dir)

    ndicoms = glob.glob(os.path.join(expected_dir, "*.dcm"))
    assert len(ndicoms) == sd.NUM_DICOMS

    sd.clean_working_dir()


def test_PrepDicom_move_dicoms_to_workdir_unzipped():

    unziped_dir = sd.unzip_t1()
    prepper = Preppers.PrepDicom(work_dir=sd.WORKING_DIR, input_file_path=unziped_dir)
    prepper.move_dicoms_to_workdir()

    expected_dir = prepper.orig_dir
    print(expected_dir)
    assert os.path.exists(expected_dir)

    ndicoms = glob.glob(os.path.join(expected_dir, "*.dcm"))
    assert len(ndicoms) == sd.NUM_DICOMS

    sd.clean_working_dir()


def test_PrepDicom_copy_dicoms_for_export():

    unziped_dir = sd.unzip_t1()
    prepper = Preppers.PrepDicom(
        work_dir=sd.WORKING_DIR, input_file_path=sd.DICOM_TEST_DATA
    )
    prepper.orig_dir = unziped_dir
    prepper.copy_dicoms_for_export()

    expected_dir = prepper.output_dir
    ndicoms = glob.glob(os.path.join(expected_dir, "*.dcm"))

    assert expected_dir != unziped_dir
    assert os.path.exists(expected_dir)
    assert len(ndicoms) == sd.NUM_DICOMS

    sd.clean_working_dir()
