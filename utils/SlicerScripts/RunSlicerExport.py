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


parser = argparse.ArgumentParser()
parser.add_argument("--dcmtk", help="use dcmtk to parse dicom (exclusive with --gdcm)", action="store_true")
parser.add_argument("--gdcm", help="use dcmtk to parse dicom (exclusive with --dcmtk)", action="store_true")
parser.add_argument("--no-quit", help="For debugging, don't exit Slicer after converting", action="store_true")
parser.add_argument("--input", help="Input DICOM directory")
parser.add_argument("--output", help="Output directory")

args = parser.parse_args()
if args.dcmtk and args.gdcm:
    raise ValueError("Cannot specify both gdcm and dcmtk")
if args.dcmtk:
    setDICOMReaderApproach('DCMTK')
if args.gdcm:
    setDICOMReaderApproach('GDCM')



print('Slicer database:  ' +str(slicer.dicomDatabase))

indexer = ctk.ctkDICOMIndexer()
dbDir = "/tmp/SlicerDB"
print("Temporary directory:  " +dbDir)
DICOMUtils.openTemporaryDatabase(dbDir)
db = slicer.dicomDatabase
indexer.addDirectory(db, args.input)
indexer.waitForImportFinished()

slicer.util.selectModule('DICOM')
# popup = slicer.modules.DICOMWidget.detailsPopup
# popup.open()

fileLists = []
for patient in db.patients():
    print("Patient: " +patient)
    for study in db.studiesForPatient(patient):
        print("Study: " +study)
        for series in db.seriesForStudy(study):
            print("Series: " +series)
            fileLists.append(db.filesForSeries(series))

print("File lists after the loop: " +str(fileLists))

nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')

if len(nodes ) >1:
    print("Input dataset resulted in more than one scalar node! Aborting.")
elif len(nodes )==0:
    print("No scalar volumes parsed from the input DICOM dataset! Aborting.")
else:
    path = os.path.join(args.output, 'volume.nrrd')
    print('Saving to ', path)
    slicer.util.saveNode(nodes[0], path)

import shutil
shutil.rmtree("/tmp/SlicerDB")

sys.exit()