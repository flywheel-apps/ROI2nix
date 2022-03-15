
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
from zipfile import ZipFile

from utils.roi_tools import InvalidDICOMFile

log = logging.getLogger(__name__)


"""
Preppers - 
Process 1 of 4
these are designed to prepare any local data for roi creation.  
This typically means the following:
    1. copy the original data (nifti or dicom) to a "working" directory.
    (/flywheel/v0/work/original_image)
    2. create a second copy of the original data into an "roi" directory
    (/flywheel/v0/work/roi_image)
    
    For dicoms, this means either unzipping a zipped file or copying in a full
    directory.  For niftis this will mean just copying the file.
    
    The "roi" image will later be modified directly to create the ROI's
    
Responsibilities:
1. prepare an "original" working directory with a copy of the data
2. prepare an "roi" working directory with a copy of the data
    
Full process:
1. Prep
2. Collect
3. Generate
4. Convert

"""

class PrepWorker(ABC):
    def __init__(self, orig_dir, output_dir, input_file_path):
        self.orig_dir = orig_dir
        self.output_dir = output_dir
        self.input_file_path = input_file_path

    @abstractmethod
    def prep(self):
        pass


@dataclass
class Prepper:
    work_dir: Path
    input_file_path: Path
    prepper: PrepWorker = None

    def __post_init__(self):
        self.output_dir = self.work_dir / "roi_image"
        self.orig_dir = self.work_dir / "original_image"
        self.prepper = self.prepper(
            self.orig_dir, self.output_dir, self.input_file_path
        )

    def prep(self):
        self.prepper.prep()





class PrepDicom(PrepWorker):
    def prep(self):
        """
        For the dicom prepper, this does the following:
        1. if the dicom is zipped, unzip to `workdir/original_image`
         - if unzipped, copy the directory
        2. make a copy of the files in `workdir/roi_image` to be processed

        Returns:

        """
        self.move_dicoms_to_workdir()
        self.copy_dicoms_for_export()

    def move_dicoms_to_workdir(self):
        # if archived, unzip dicom into work/dicom/
        self.orig_dir.mkdir(parents=True, exist_ok=True)
        if self.input_file_path.as_posix().endswith(".zip"):
            # Unzip, pulling any nested files to a top directory.
            dicom_zip = ZipFile(self.input_file_path)
            for member in dicom_zip.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
                # copy file (taken from zipfile's extract)
                source = dicom_zip.open(member)
                target = open(os.path.join(self.orig_dir, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
            # dicom_zip.extractall(path=dicom_dir)
        else:
            shutil.copy(self.input_file_path, self.orig_dir)

    def copy_dicoms_for_export(self):
        try:
            # Copy dicoms to a new directory
            if not os.path.exists(self.output_dir):
                shutil.copytree(self.orig_dir, self.output_dir)
        except Exception as e:
            log.exception(e)
            error_message = (
                "An invalid dicom file was encountered. "
                '"SeriesInstanceUID", "InstanceNumber", '
                '"ImageOrientationPatient", or "ImagePositionPatient"'
                " may be missing from the DICOM series."
            )
            raise InvalidDICOMFile(error_message)


class PrepNifti(PrepWorker):
    def prep(self):
        pass
