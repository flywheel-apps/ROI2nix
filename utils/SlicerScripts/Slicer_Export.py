import argparse
import glob
import os
import DICOMScalarVolumePlugin as dsvp
import sys

def get_file_list(input_file):

    if not os.path.exists(input_file):
        raise Exception("file or folder {input_file}".format(input_file=input_file))

    if os.path.isdir(input_file):
        working_dir = input_file
        extensions = ['.dcm', '.DCM', '.ima', '.IMA']

    elif os.path.isfile(input_file):
        stripped_file, ext = os.path.splitext(input_file)
        working_dir = os.path.dirname(stripped_file)
        extensions = [ext]

    for ex in extensions:
        glob_string = os.path.join(working_dir, '*{}'.format(ex))
        file_list = glob.glob(glob_string)

        if len(file_list) > 0:
            break

        else:
            print("no files found in {working_dir} with extension {ext}".format(working_dir=working_dir, ext=ex))

    if len(file_list) == 0:
        raise Exception('No files found in directory {working_dir}'.format(working_dir=working_dir))

    file_list.sort()

    return file_list



parser = argparse.ArgumentParser()
parser.add_argument("--dcmtk", help="use dcmtk to parse dicom (exclusive with --gdcm)", action="store_true")
parser.add_argument("--gdcm", help="use dcmtk to parse dicom (exclusive with --dcmtk)", action="store_true")
parser.add_argument("--archetype", help="Load files in the traditional Slicer manner using the volume logic helper class and the vtkITK archetype helper code", action="store_true")
parser.add_argument("--input", help="directory with dicoms or one file in the series.")
parser.add_argument("--output", help="Output directory")
parser.add_argument("--filename", help="output file name")
parser.add_argument("--extension", help="extension")
args = parser.parse_args()

# XOR on three conditions, only one can be true at a time
if not (args.dcmtk ^ args.gdcm) | (args.gdcm ^ args.archetype):
    raise ValueError("Cannot specify both gdcm and dcmtk and archeype")

if args.dcmtk:
    method = "DCMTK"
elif args.gdcm:
    method = "GDCM"
elif args.archetype:
    method = "ARCH"

files = get_file_list(args.input)

loader = dsvp.DICOMScalarVolumePluginClass()

if args.archetype:
    node = loader.loadFilesWithArchetype(files, 'node')
else:
    node = loader.loadFilesWithSeriesReader(method, files, 'node')

output_file = os.path.join(args.output, "{name}.{ext}".format(name=args.filename, ext=args.extension))

slicer.util.saveNode(node, output_file)
sys.exit()

