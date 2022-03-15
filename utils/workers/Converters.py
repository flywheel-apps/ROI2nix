
from abc import ABC, abstractmethod
from dataclasses import dataclass
import glob
import logging
import os
from pathlib import Path
import pydicom
import re
import shutil
import subprocess as sp
import traceback
import unicodedata

from utils.objects.Conversion import ConversionType, NIFTI_TYPE, NRRD_TYPE

log = logging.getLogger()



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

SLICER_SCRIPT = f"{os.environ['SCRIPT_DIR']}/RunSlicerExport.py"


class ConvertWorker(ABC):
    def __init__(self, orig_dir, roi_dir, output_dir, ext=NIFTI_TYPE):
        self.orig_dir = orig_dir
        self.roi_dir = roi_dir
        self.output_dir = output_dir
        self.ext = ext
        self.additional_args = "" # TODO: These all need to accept additional command options from a config string


    @abstractmethod
    def convert(self, output_filename):
        pass


@dataclass
class Converter:
    orig_dir: Path
    roi_dir: Path
    combine: bool
    bitmask: bool
    output_dir: Path
    conversion: ConversionType
    converter: ConvertWorker = None

    def __post_init__(self):
        self.converter = self.converter(self.orig_dir, self.roi_dir, self.output_dir, self.conversion.ext)

    def convert_dir(self, output_filename):
        self.converter.convert(output_filename)


class dcm2niix(ConvertWorker):
    def run_command(self, command):
        pr = sp.Popen(command)
        pr.wait()

    def convert(self, output_filename):
        command = self.make_command(output_filename)
        self.run_command(command)

    def make_command(self, output_filename):
        nrrd_cmd = ['']
        if self.converter.ext == NRRD_TYPE:
            nrrd_cmd = ['-e', 'y']

        command = [
            "dcm2niix",
            "-o",
            self.output_dir.as_posix(),
            "-f",
            output_filename,
            "-b",
            "n",
            *nrrd_cmd,
            self.roi_dir.as_posix(),
        ]
        command = [c for c in command if c]
        print(' '.join(command))
        return command


class slicer_dcmtk(ConvertWorker):
    def run_command(self, command):
        pr = sp.Popen(command)
        pr.wait()

    def convert(self, output_filename):
        command = self.make_command(output_filename)
        self.run_command(command)

    def make_command(self, output_filename):
        output_filename = output_filename+f'.{self.ext}'

        command = ["xvfb-run",
            "Slicer",
            "--python-script",
            SLICER_SCRIPT,
            "--dcmtk",
            "--input",
            self.roi_dir.as_posix(),
            "--output",
            self.output_dir.as_posix(),
            "--filename",
            output_filename
        ]

        # output=os.path.join(self.output_dir,output_filename+'.{}'.format(self.ext))
        # first_file = glob.glob(os.path.join(self.roi_dir.as_posix(),'*.dcm'))
        # first_file.sort()
        # first_file = first_file[0]
        #
        # pycode="node=slicer.util.loadVolume('"+first_file+"');slicer.util.saveNode(node, '"+output+"');exit()"
        #
        # command = ["xvfb-run",
        #            "Slicer","--no-main-window",
        #            "--python-code",
        #            pycode
        # ]
        print(' '.join(command))
        return command

class slicer_gdcm(ConvertWorker):
    def run_command(self, command):
        pr = sp.Popen(command)
        pr.wait()

    def convert(self, output_filename):
        command = self.make_command(output_filename)
        self.run_command(command)

    def make_command(self, output_filename):
        output_filename = output_filename+f'.{self.ext}'

        command = ["xvfb-run",
            "Slicer",
            "--python-script",
            SLICER_SCRIPT,
            "--gdcm",
            "--input",
            self.roi_dir.as_posix(),
            "--output",
            self.output_dir.as_posix(),
            "--filename",
            output_filename
        ]
        return command

class slicer_arch(ConvertWorker):
    def run_command(self, command):
        pr = sp.Popen(command)
        pr.wait()

    def convert(self, output_filename):
        command = self.make_command(output_filename)
        self.run_command(command)

    def make_command(self, output_filename):
        output_filename = output_filename+f'.{self.ext}'

        command = ["xvfb-run",
            "Slicer",
            "--python-script",
            SLICER_SCRIPT,
            "--archetype",
            "--input",
            self.roi_dir.as_posix(),
            "--output",
            self.output_dir.as_posix(),
            "--filename",
            output_filename,

        ]
        print(' '.join(command))
        return command

class plastimatch(ConvertWorker):
    def run_command(self, command):
        pr = sp.Popen(command)
        pr.wait()

    def convert(self, output_filename):
        command = self.make_command(output_filename)
        self.run_command(command)

    def make_command(self, output_filename):

        output_filename = os.path.join(self.output_dir.as_posix(), output_filename+f'.{self.ext}')
        command = ["plastimatch",
                   "convert",
                   "--input",
                   self.roi_dir.as_posix(),
                   "--output-img",
                   output_filename
        ]
        print(' '.join(command))
        return command

class dicom2nifti(ConvertWorker):
    def run_command(self, command):
        pr = sp.Popen(command)
        pr.wait()

    def convert(self, output_filename):
        command = self.make_command()
        self.run_command(command)
        self.rename_dicom2nifti_output(output_filename)


    def make_command(self):


        command = ["dicom2nifti",
                   self.roi_dir.as_posix(),
                   self.output_dir.as_posix()
        ]
        print(' '.join(command))

        return command

    def rename_dicom2nifti_output(self, output_filename):

        orig_output_basename = self.guess_dicom2nifti_outputname()
        orig_output_basename += ".*"
        glob_string = self.output_dir/orig_output_basename
        found_file = glob.glob(glob_string.as_posix())
        if not found_file:
            log.error('error converting with dicom2nifti')
            raise Exception('error converting with dicom2nifti')

        found_file = found_file[0]
        ext = found_file[found_file.find('.nii'):]

        output_filename = os.path.join(self.output_dir.as_posix(), output_filename+ext)
        log.debug(f'found {found_file}, renaming to {output_filename}')
        shutil.move(found_file, output_filename)


    def guess_dicom2nifti_outputname(self):
        first_dicom = glob.glob(os.path.join(self.roi_dir.as_posix(), '*'))[0]
        dicom_input = pydicom.read_file(first_dicom)


        base_filename = ""
        if 'SeriesNumber' in dicom_input:
            base_filename = _remove_accents('%s' % dicom_input.SeriesNumber)
            if 'SeriesDescription' in dicom_input:
                base_filename = _remove_accents('%s_%s' % (base_filename,
                                                           dicom_input.SeriesDescription))
            elif 'SequenceName' in dicom_input:
                base_filename = _remove_accents('%s_%s' % (base_filename,
                                                           dicom_input.SequenceName))
            elif 'ProtocolName' in dicom_input:
                base_filename = _remove_accents('%s_%s' % (base_filename,
                                                           dicom_input.ProtocolName))
        else:
            base_filename = _remove_accents(dicom_input.SeriesInstanceUID)

        return base_filename

def _remove_accents(unicode_filename):
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    # noinspection PyBroadException
    try:
        unicode_filename = unicode_filename.replace(" ", "_")
        cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore').decode('ASCII')

        cleaned_filename = re.sub(r'[^\w\s-]', '', cleaned_filename.strip().lower())
        cleaned_filename = re.sub(r'[-\s]+', '-', cleaned_filename)

        return cleaned_filename
    except:
        traceback.print_exc()
        return unicode_filename