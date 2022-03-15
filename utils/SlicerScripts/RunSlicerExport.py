import os
import sys
from DICOMLib import DICOMUtils
import argparse
import DICOMScalarVolumePlugin


def setDICOMReaderApproach(approach):
    approaches = DICOMScalarVolumePlugin.DICOMScalarVolumePluginClass.readerApproaches()
    if approach not in approaches:
        raise ValueError("Unknown dicom approach: %s\nValid options are: %s" % (approach, approaches))
    approachIndex = approaches.index(approach)
    settings = qt.QSettings()
    settings.setValue('DICOM/ScalarVolume/ReaderApproach', approachIndex)

# node=slicer.util.loadVolume('/flywheel/v0/scrap/1-01.dcm')
# slicer.util.saveNode(node, '/flywheel/v0/output/ROI_Potato_T2-haste_ax.nii')
# exit()

parser = argparse.ArgumentParser()
parser.add_argument("--dcmtk", help="use dcmtk to parse dicom (exclusive with --dcmtk)", action="store_true")
parser.add_argument("--gdcm", help="use gdcm to parse dicom (exclusive with --gdcm)", action="store_true")
parser.add_argument("--archetype", help="use archetype to parse dicom (exclusive with --archetype)", action="store_true")
parser.add_argument("--input", help="Input DICOM directory")
parser.add_argument("--output", help="Output directory")
parser.add_argument("--filename", help="File name")
args = parser.parse_args()


# XOR on three conditions, only one can be true at a time
if not (args.dcmtk ^ args.gdcm) | (args.gdcm ^ args.archetype):
    raise ValueError("Cannot specify both gdcm and dcmtk and archetype")

if args.dcmtk:
    setDICOMReaderApproach('DCMTK')
if args.gdcm:
    setDICOMReaderApproach('GDCM')
if args.archetype:
    setDICOMReaderApproach('Archetype')


indexer = ctk.ctkDICOMIndexer()
dbDir = "/tmp/SlicerDB"
print("Temporary directory:  " +dbDir)
DICOMUtils.openTemporaryDatabase(dbDir)
db = slicer.dicomDatabase
print("indexing {}".format(args.input))
indexer.addDirectory(db, args.input)
indexer.waitForImportFinished()

slicer.util.selectModule('DICOM')
popup = slicer.modules.DICOMWidget.browserWidget
popup.open()

fileLists = []
for patient in db.patients():
    print("Patient:"+patient)
    for study in db.studiesForPatient(patient):
        print("Study:"+study)
        for series in db.seriesForStudy(study):
            print("Series:"+series)
            fileLists.append(db.filesForSeries(series))
print("File lists after the loop:"+str(fileLists))
popup.fileLists = fileLists

popup.examineForLoading()
popup.organizeLoadables()
popup.loadCheckedLoadables()

nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')

if len(nodes ) >1:
    print("Input dataset resulted in more than one scalar node! Aborting.")
elif len(nodes )==0:
    print("No scalar volumes parsed from the input DICOM dataset! Aborting.")
else:
    path = os.path.join(args.output, args.filename)
    print('Saving to ', path)
    slicer.util.saveNode(nodes[0], path)

import shutil
shutil.rmtree("/tmp/SlicerDB")

sys.exit()