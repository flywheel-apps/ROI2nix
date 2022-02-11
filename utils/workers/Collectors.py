from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from zipfile import ZipFile
import os
import shutil
import logging
import pydicom
from utils.utils import InvalidConversion, InvalidDICOMFile, InvalidROIError
from flywheel import Client, FileEntry
from collections import OrderedDict
import glob
import numpy as np
import utils.utils as utils
import re
import subprocess as sp
from utils.Labels import RoiLabel
from scipy import stats

log = logging.getLogger(__name__)




@dataclass
class LabelCollector():
    fw_client: Client
    file_object: FileEntry
    collector: CollWorker: None

    def collect_rois(self):
        self.collector.collect()



class PrepWorker(ABC):
    def __init__(self, orig_dir, output_dir, input_file_path):
        self.orig_dir = orig_dir
        self.output_dir = output_dir
        self.input_file_path = input_file_path

    @abstractmethod
    def prep(self):
        pass