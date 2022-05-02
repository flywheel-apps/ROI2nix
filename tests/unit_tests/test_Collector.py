import logging
import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from utils.workers import Collectors
from tests import setup_data as sd

log = logging.getLogger(__name__)


def test_BaseCollector_get_ohif_info_from_file():

    file_object = MagicMock()
    flywheel_file = MagicMock()
    flywheel_file.name = "mock_file_name"
    flywheel_file["info"].get.return_value = "mock_file_info"

    file_object.flywheel_file = flywheel_file

    collector = Collectors.DicomRoiCollector(
        fw_client=MagicMock(), orig_dir="", file_object=file_object
    )

    collector.get_ohif_info()
    print(collector.ohifviewer_info)
    assert collector.ohifviewer_info == "mock_file_info"
    collector.fw_client.get_session.assert_not_called()
    collector.file_object.flywheel_file["info"].get.assert_called_with("ohifViewer")


def test_BaseCollector_get_ohif_info_from_session():

    file_object = MagicMock()
    flywheel_file = MagicMock()
    flywheel_file.name = "mock_file_name"
    flywheel_file.get.return_value = False
    file_object.flywheel_file = flywheel_file

    fw_client = MagicMock()

    session = MagicMock()
    session.label = "mock_session_label"
    session.info = {"ohifViewer": "mock_session_info"}

    fw_client.get_session.return_value = session

    collector = Collectors.DicomRoiCollector(
        fw_client=fw_client, orig_dir="", file_object=file_object
    )

    collector.get_ohif_info()
    print(collector.ohifviewer_info)
    assert collector.ohifviewer_info == "mock_session_info"
    collector.fw_client.get_session.assert_called_once()
    collector.file_object["info"].assert_not_called()


def test_BaseCollector_factory():

    tests = [
        ("dicom", Collectors.DicomRoiCollector),
        ("nifti", Collectors.NiftiRoiCollector),
    ]

    for test in tests:
        test_type, test_class = test
        collector = Collectors.BaseCollector.factory(
            type_=test_type, fw_client=None, file_object=None, orig_dir=None
        )
        assert isinstance(collector, test_class)

    with pytest.raises(NotImplementedError):
        Collectors.BaseCollector.factory(
            type_="invalid", fw_client=None, file_object=None, orig_dir=None
        )


def test_DicomRoiCollector_collect():
    # Nothing actually needs to be tested here.
    # get_ohif_info, get_current_study_series_uid, and
    # identify_rois_on_image are all that's called and they're
    # tested elsewhere.

    pass


def test_DicomRoiCollector_get_current_study_series_uid():

    orig_dir = sd.unzip_t1()

    fw_client = MagicMock()
    file_object = MagicMock()
    collector = Collectors.DicomRoiCollector(
        fw_client=fw_client, orig_dir=orig_dir, file_object=file_object
    )

    studyInstance, seriesInstance = collector.get_current_study_series_uid()
    assert studyInstance == sd.T2_PHANTOM_STUDY_UID
    assert seriesInstance == sd.T2_PHANTOM_SERIES_UID
    sd.clean_working_dir()


def test_DicomRoiCollector_identify_rois_on_image():

    fw_client = MagicMock()
    file_object = MagicMock()
    collector = Collectors.DicomRoiCollector(
        fw_client=fw_client, orig_dir="", file_object=file_object
    )

    test_ohif_info = sd.get_dicom_raw_roi_info()
    collector.ohifviewer_info = test_ohif_info
    collector.identify_rois_on_image(sd.T2_PHANTOM_STUDY_UID, sd.T2_PHANTOM_SERIES_UID)
    new_ohif_info = collector.ohifviewer_info

    assert test_ohif_info != new_ohif_info

    for roi_type in new_ohif_info:
        for roi in new_ohif_info[roi_type]:
            assert roi["SeriesInstanceUID"] == sd.T2_PHANTOM_SERIES_UID

    processed_template = sd.get_dicom_processed_roi_info()
    assert new_ohif_info == processed_template


def test_cleanup():
    sd.clean_working_dir()
