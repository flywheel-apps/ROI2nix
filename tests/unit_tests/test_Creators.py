from collections import OrderedDict
import copy
import logging
import numpy as np
import os
from pathlib import Path
import pytest
import pydicom
from unittest.mock import MagicMock
from unittest.mock import patch
from tests import setup_data as sd
from fw_file.dicom import DICOMCollection
from utils.objects.Labels import RoiLabel


os.environ["SCRIPT_DIR"] = os.path.join(sd.cwd, "utils/SlicerScripts")
from utils.workers import Creators


log = logging.getLogger(__name__)


def test_BaseCreator_factory():

    tests = [
        ("dicom", Creators.DicomCreator),
        ("nifti-notimplemented", Creators.NiftiCreator),
    ]

    for test in tests:
        test_type, test_class = test
        creator = Creators.BaseCreator.factory(
            type_=test_type,
            orig_dir=None,
            roi_dir=None,
            output_dir=None,
            base_file_name=None,
            combine=None,
            bitmask=None,
            converter=None,
        )
        assert isinstance(creator, test_class)

    test_type = "random_type"
    with pytest.raises(NotImplementedError):
        creator = Creators.BaseCreator.factory(
            type_=test_type,
            orig_dir=None,
            roi_dir=None,
            output_dir=None,
            base_file_name=None,
            combine=None,
            bitmask=None,
            converter=None,
        )


def test_BaseCreator_get_labels():
    test_type = "dicom"
    metadata_json = sd.get_dicom_processed_roi_info()
    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=None,
        roi_dir=None,
        output_dir=None,
        base_file_name=None,
        combine=None,
        bitmask=None,
        converter=None,
    )
    labels = creator.get_labels(metadata_json)
    print(labels)
    assert labels == OrderedDict(
        [
            (
                "Lesion",
                RoiLabel(
                    label="Lesion",
                    index=1,
                    color="#f44336",
                    RGB=[244, 67, 54, 0.2],
                    num_voxels=0,
                    volume=0.0,
                ),
            ),
            (
                "Potato",
                RoiLabel(
                    label="Potato",
                    index=2,
                    color="#cddc39",
                    RGB=[205, 220, 57, 0.2],
                    num_voxels=0,
                    volume=0.0,
                ),
            ),
            (
                "Doggy",
                RoiLabel(
                    label="Doggy",
                    index=4,
                    color="#4caf50",
                    RGB=[76, 175, 80, 0.2],
                    num_voxels=0,
                    volume=0.0,
                ),
            ),
        ]
    )


def test_DicomCreator_generate_name():
    test_type = "dicom"
    base_file_name = "base_file_name"
    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=None,
        roi_dir=None,
        output_dir=None,
        base_file_name=base_file_name,
        combine=None,
        bitmask=None,
        converter=None,
    )

    unchanged_label = "good_label_name"
    changed_label = "bad %label -name"

    gen_unchanged = creator.generate_name(unchanged_label, False)
    gen_changed = creator.generate_name(changed_label, False)
    test_all = creator.generate_name(changed_label, True)

    output_filename = "ROI_" + unchanged_label + "_" + base_file_name
    assert gen_unchanged == output_filename

    output_filename = "ROI_" + "bad_label_name" + "_" + base_file_name
    assert gen_changed == output_filename

    output_filename = "ROI_" + "ALL" + "_" + base_file_name
    assert test_all == output_filename

    label = "label"

    strip_nii = "name.nii"
    strip_niigz = "name.nii.gz"
    strip_dicomzip = "name.dicom.zip"
    strip_zip = "name.zip"

    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=None,
        roi_dir=None,
        output_dir=None,
        base_file_name=strip_nii,
        combine=None,
        bitmask=None,
        converter=None,
    )
    gen_nii = creator.generate_name(label, False)

    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=None,
        roi_dir=None,
        output_dir=None,
        base_file_name=strip_niigz,
        combine=None,
        bitmask=None,
        converter=None,
    )
    gen_niigz = creator.generate_name(label, False)

    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=None,
        roi_dir=None,
        output_dir=None,
        base_file_name=strip_dicomzip,
        combine=None,
        bitmask=None,
        converter=None,
    )
    gen_dicomzip = creator.generate_name(label, False)

    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=None,
        roi_dir=None,
        output_dir=None,
        base_file_name=strip_zip,
        combine=None,
        bitmask=None,
        converter=None,
    )
    gen_zip = creator.generate_name(label, False)

    output_filename = "ROI_" + label + "_name"
    assert gen_nii == output_filename

    output_filename = "ROI_" + label + "_name"
    assert gen_niigz == output_filename

    output_filename = "ROI_" + label + "_name"
    assert gen_dicomzip == output_filename

    output_filename = "ROI_" + label + "_name"
    assert gen_zip == output_filename


def test_DicomCreator_get_dicoms():

    dicom_dir = sd.unzip_t1()
    test_type = "dicom"
    base_file_name = "T2_Phantom"
    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=dicom_dir,
        roi_dir=None,
        output_dir=None,
        base_file_name=base_file_name,
        combine=None,
        bitmask=None,
        converter=None,
    )

    creator.get_dicoms()
    assert creator.shape == [512, 368, 10]

    sd.clean_working_dir()


def test_DicomCreator_get_affine():

    dicom_dir = sd.unzip_t1()
    test_type = "dicom"
    base_file_name = "T2_Phantom"
    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=dicom_dir,
        roi_dir=None,
        output_dir=None,
        base_file_name=base_file_name,
        combine=None,
        bitmask=None,
        converter=None,
    )

    creator.get_dicoms()
    affine = creator.get_affine()
    my_affine = np.array([[4.0, 0, 0], [0, 0.3515625, 0], [0, 0, 0.3515625]])
    assert np.array_equal(affine, my_affine)


def test_DicomCreator_fill_roi_dicom_slice():

    dicom_dir = sd.unzip_t1()
    test_type = "dicom"
    base_file_name = "T2_Phantom"
    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=dicom_dir,
        roi_dir=None,
        output_dir=None,
        base_file_name=base_file_name,
        combine=None,
        bitmask=None,
        converter=None,
    )
    creator.get_dicoms()
    dicoms = creator.dicoms
    dicom_sops = dicoms.bulk_get("SOPInstanceUID")
    sop = dicom_sops[0]
    slice = dicom_sops.index(sop)

    roi_handles = {"start": {"x": 0, "y": 0}, "end": {"x": 20, "y": 25}}

    data = np.zeros(creator.shape)
    match_data = copy.deepcopy(data)
    match_data[
        roi_handles["start"]["y"] : roi_handles["end"]["y"] + 1,
        roi_handles["start"]["x"] : roi_handles["end"]["x"] + 1,
        slice,
    ] = 1

    data = creator.fill_roi_dicom_slice(
        data=data,
        sop=sop,
        roi_handles=roi_handles,
        roi_type="RectangleRoi",
        dicoms=dicoms,
    )

    assert np.array_equal(match_data, data)


def test_DicomCreator_label2data():

    dicom_dir = sd.unzip_t1()
    test_type = "dicom"
    base_file_name = "T2_Phantom"

    metadata_json = sd.get_dicom_processed_roi_info()
    label = "Lesion"

    for roi_type in metadata_json:
        for roi in metadata_json[roi_type]:
            if roi.get("location") == label:

                ar = np.zeros((0, 0, 0), dtype=bool)
                sop = roi["SOPInstanceUID"]
                handles = roi["handles"]
                dicoms = {}
                returnroi_type = roi_type

    with patch("utils.workers.Creators.DicomCreator.fill_roi_dicom_slice", MagicMock()):

        creator = Creators.BaseCreator.factory(
            type_=test_type,
            orig_dir=dicom_dir,
            roi_dir=None,
            output_dir=None,
            base_file_name=base_file_name,
            combine=None,
            bitmask=None,
            converter=None,
        )

        data = creator.label2data(label, metadata_json)
        print(creator.fill_roi_dicom_slice.call_args_list)
        creator.fill_roi_dicom_slice.assert_called_once()
        # Note: for some reason this doens't work with "assert_called_with"
        creator.fill_roi_dicom_slice.call_args_list[0] == (
            (ar, sop, handles),
            {"dicoms": {}, "roi_type": returnroi_type},
        )


def test_DicomCreator_set_bit_level(caplog):
    type = "dicom"
    creator = Creators.BaseCreator.factory(
        type_=type,
        orig_dir=None,
        roi_dir=None,
        output_dir=None,
        base_file_name=None,
        combine=False,
        bitmask=False,
        converter=None,
    )
    labels = list(range(10))
    creator.set_bit_level(labels)
    assert creator.bits == 8
    assert creator.dtype == np.uint8

    creator.combine = True
    creator.set_bit_level(labels)
    assert creator.bits == 16
    assert creator.dtype == np.uint16

    creator.bits = 0
    creator.dtype = 0

    creator.combine = False
    creator.bitmask = True
    creator.set_bit_level(labels)
    assert creator.bits == 16
    assert creator.dtype == np.uint16

    labels = list(range(20))
    creator.set_bit_level(labels)
    assert creator.bits == 32
    assert creator.dtype == np.uint32

    labels = list(range(40))

    with caplog.at_level(logging.DEBUG):
        creator.set_bit_level(labels)

    assert "Due to the maximum integer length" in caplog.text
    sd.clean_working_dir()


def test_DicomCreator_save_to_roi_dir():
    dicom_dir = sd.unzip_t1()
    test_type = "dicom"
    base_file_name = "T2_Phantom"
    roi_dir = Path(sd.WORKING_DIR) / "roi_dir"

    if not os.path.exists(roi_dir):
        os.mkdir(roi_dir)

    creator = Creators.BaseCreator.factory(
        type_=test_type,
        orig_dir=dicom_dir,
        roi_dir=roi_dir,
        output_dir=None,
        base_file_name=base_file_name,
        combine=None,
        bitmask=None,
        converter=None,
    )
    creator.get_dicoms()

    data = np.random.rand(creator.shape[0], creator.shape[1], creator.shape[2]) * 100
    data = data.astype(creator.dtype)
    creator.save_to_roi_dir(data)

    saved_dicoms = DICOMCollection.from_dir(roi_dir)
    for dcm in saved_dicoms:
        dicom_file = Path(dcm.filepath)
        dicom_data = pydicom.read_file(dicom_file)
        num = dcm.InstanceNumber
        flat_data = data[:, :, num - 1]
        flat_data = flat_data.flatten()
        flat_data = flat_data.tobytes()

        assert dicom_data.get("PixelData") == flat_data

    sd.clean_working_dir()
