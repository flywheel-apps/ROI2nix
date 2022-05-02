
from fw_file.dicom import DICOMCollection
import gzip
import json
import os
import shutil

cwd = os.path.dirname(__file__)
DICOM_TEST_DATA = os.path.join(cwd, "test_data/DICOM/T2_Phantom.dicom.zip")
NIFTI_TEST_DATA = os.path.join(cwd, "test_data/NIFTI/T2_Phantom.nii.gz")
WORKING_DIR = os.path.join(cwd, "test_data/DICOM/test_work")

T2_PHANTOM_SERIES_UID='1.3.12.2.1107.5.2.43.166011.2019022718531497064940613.0.0.0'
T2_PHANTOM_STUDY_UID='1.3.12.2.1107.5.2.43.166011.30000019021819133110600000088'

DICOM_DIR = T2_PHANTOM_SERIES_UID+".dicom"
NUM_DICOMS = 10

def setup_nifti(nifti_name = "", unzip=False):
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    if not nifti_name:
        nifti_name = "T2_Phantom.nii.gz"

    new_out = os.path.join(WORKING_DIR, nifti_name)

    shutil.copy(NIFTI_TEST_DATA, new_out)

    if not unzip:
        return new_out

    unzipped_out = os.path.splitext(new_out)[0]
    with gzip.open(new_out, 'rb') as f_in:
        with open(unzipped_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(new_out)
    return unzipped_out




def get_zipped_t1():

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    new_dest = os.path.join(WORKING_DIR,"T2_Phantom.dicom.zip")
    shutil.copy(DICOM_TEST_DATA, new_dest)
    return new_dest

def unzip_t1():
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    dcms = DICOMCollection.from_zip(DICOM_TEST_DATA)
    dicom_dir = os.path.join(WORKING_DIR, DICOM_DIR)
    dcms.to_dir(dicom_dir)

    return dicom_dir

def get_dicom_raw_roi_info():

    metadata_path = os.path.join(cwd,'test_data','metadata','test_ohif_info.json')

    with open(metadata_path, 'r') as meta_json:
        test_ohif_info = json.load(meta_json)

    return test_ohif_info


def get_dicom_processed_roi_info():

    metadata_path = os.path.join(cwd,'test_data','metadata','test_processed_ohif_info.json')

    with open(metadata_path, 'r') as meta_json:
        test_ohif_info = json.load(meta_json)

    return test_ohif_info





def clean_working_dir():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
