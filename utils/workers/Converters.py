from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from zipfile import ZipFile
import os
import shutil
import logging
import pydicom
from utils.utils import InvalidConversion, InvalidDICOMFile, InvalidROIError
from utils.workers import Converters
from collections import OrderedDict
import glob
import numpy as np
import utils.utils as utils
import re
import subprocess as sp
from utils.Labels import RoiLabel
from scipy import stats


"""
Converter -
Process 4 of 4

Responsible for converting the working roi directory from one format to another.
This will undoubtedly become more and more complicated as more things get added...
For example, nifti to dicom will require a lot of affine information and stuff...
If we can't directly recover that data from the original file, then things will
have to get passed in which will increase complexity.  

For now I'm focusing on dicom to other formats.  Future problems are for future David.

I learned this technique to kind of split up responsibilities, and when I read the tutorials
and watch videos of simple examples it seems so straightforward and good.  Then I try to 
implement it, and i have all these cases where thigns from one part of the code depend 
on things from other parts of the code, and it just gets complicated.  So while I started off
with the intention to simplify, I always end up with this monstrosity of a code block by the 
end, and I wonder if I actually succeeded or not.  This happens a lot.  
But I do think I'm getting better, idk. 


Full process:
1. Prep
2. Collect
3. Generate
4. Convert

"""


@dataclass
class Converter():
    orig_dir: Path
    roi_dir: Path
    combine: bool
    bitmask: bool
    output_dir: Path
    labels: dict
    converter: ConvertWorker = None

    def __post_init__(self):
        self.converter = self.converter(self.orig_dir, self.roi_dir, self.output_dir)

    def convert_dir(self, output_filename):
        self.converter.convert(output_filename)



class ConvertWorker(ABC):
    def __init__(self, orig_dir, roi_dir, output_dir):
        self.orig_dir = orig_dir
        self.roi_dir = roi_dir
        self.output_dir = output_dir

    def convert(self, output_filename):
        command = self.make_command(output_filename)
        self.run_command(command)

    def run_command(self, command):
        pr = sp.Popen(command)
        pr.wait()

    @abstractmethod
    def make_command(self, output_filename):
        pass

class dcm2niix_nifti(ConvertWorker):
    def make_command(self, output_filename):
        command = [
            "dcm2niix",
            "-o",
            self.output_dir,
            "-f",
            output_filename,
            "-b",
            "n",
            self.roi_dir,
        ]
        return command

class dcm2niix_nrrd(ConvertWorker):
    def make_command(self, output_filename):
        command = [
            "dcm2niix",
            "-o",
            self.output_dir,
            "-f",
            output_filename,
            "-e",
            "y",
            "-b",
            "n",
            self.roi_dir,
        ]
        return command

class slicer_nifti(ConvertWorker):
    def make_command(self, output_filename):
        command = [
            "dcm2niix",
            "-o",
            self.output_dir,
            "-f",
            output_filename,
            "-e",
            "y",
            "-b",
            "n",
            self.roi_dir,
        ]
        return command

class slicer_nrrd(ConvertWorker):
    def make_command(self, output_filename):
        command = [
            "dcm2niix",
            "-o",
            self.output_dir,
            "-f",
            output_filename,
            "-e",
            "y",
            "-b",
            "n",
            self.roi_dir,
        ]
        return command

