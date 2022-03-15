import os
import sys
from DICOMLib import DICOMUtils
import argparse
import DICOMScalarVolumePlugin as dsvp


def setDICOMReaderApproach(approach):
    approaches = DICOMScalarVolumePlugin.DICOMScalarVolumePluginClass.readerApproaches()
    if approach not in approaches:
        raise ValueError("Unknown dicom approach: %s\nValid options are: %s" % (approach, approaches))
    approachIndex = approaches.index(approach)
    settings = qt.QSettings()
    settings.setValue('DICOM/ScalarVolume/ReaderApproach', approachIndex)

node=slicer.util.loadVolume('/flywheel/v0/scrap/1-01.dcm')
slicer.util.saveNode(node, '/flywheel/v0/output/ROI_Potato_T2-haste_ax.nii')
exit()

parser = argparse.ArgumentParser()
parser.add_argument("--dcmtk", help="use dcmtk to parse dicom (exclusive with --gdcm)", action="store_true")
parser.add_argument("--gdcm", help="use dcmtk to parse dicom (exclusive with --dcmtk)", action="store_true")
parser.add_argument("--input", help="Input DICOM directory")
parser.add_argument("--output", help="Output directory")
parser.add_argument("--filename", help="File name")
args = parser.parse_args()

if args.dcmtk and args.gdcm:
    raise ValueError("Cannot specify both gdcm and dcmtk")
if args.dcmtk:
    setDICOMReaderApproach('DCMTK')
if args.gdcm:
    setDICOMReaderApproach('GDCM')


indexer = ctk.ctkDICOMIndexer()
dbDir = "/tmp/SlicerDB"
print("Temporary directory:  " +dbDir)
DICOMUtils.openTemporaryDatabase(dbDir)
db = slicer.dicomDatabase
print("indexing {}".format(args.input))
indexer.addDirectory(db, args.input)
indexer.waitForImportFinished()

slicer.util.selectModule('DICOM')

fileLists = []
for patient in db.patients():
    print("Patient:"+patient)
    for study in db.studiesForPatient(patient):
        print("Study:"+study)
        for series in db.seriesForStudy(study):
            print("Series:"+series)
            fileLists.append(db.filesForSeries(series))


for patientUID in db.patients():
    loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
node = nodes[0]

referenceVolumeNode.GetOrigin()
niftiVolumeNody.SetIJKToRASDirectionMatrix(vtkmat)
vtkmat=vtkMatrixFromArray(np.eye(4))
o=referenceVolumeNode.GetIJKToRASDirectionMatrix(vtkmat)





import shutil
shutil.rmtree("/tmp/SlicerDB")

sys.exit()