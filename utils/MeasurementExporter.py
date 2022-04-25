import logging

from utils.objects.Files import FileObject
from utils.objects import Conversion
from utils.workers import Preppers, Collectors, Converters, Creators

log = logging.getLogger(__name__)


class MeasurementExport:
    """Exports measurements from files on flywheel"""

    def __init__(
        self,
        fw_client,
        fw_file,
        work_dir,
        output_dir,
        input_file_path,
        dest_file_type,
        combine,
        bitmask,
        method,
    ):

        self.file_object = FileObject(input_file_path, fw_file)
        self.orig_file_type = self.file_object.file_type

        # Conversion object is like a pre-check of valid input/output/conversion software types
        self.conversion = self.generate_conversion(
            orig_file_type=self.file_object.file_type,
            dest_file_type=dest_file_type,
            method=method,
        )
        self.prepper = self.generate_prepper(
            work_dir=work_dir,
            input_file_path=input_file_path,
            orig_file_type=self.orig_file_type,
        )

        print(self.prepper)
        self.collector = self.generate_collector(
            fw_client=fw_client,
            orig_dir=self.prepper.orig_dir,
            file_object=self.file_object,
            orig_file_type=self.orig_file_type,
        )
        self.converter = self.generate_converter(
            conversion=self.conversion,
            orig_dir=self.prepper.orig_dir,
            roi_dir=self.prepper.output_dir,
            combine=combine,
            bitmask=bitmask,
            output_dir=output_dir,
        )
        self.creator = self.generate_creator(
            converter=self.converter,
            file_object=self.file_object,
            orig_dir=self.prepper.orig_dir,
            roi_dir=self.prepper.output_dir,
            combine=combine,
            bitmask=bitmask,
            output_dir=output_dir,
        )

    def get_affine(self):
        return self.collector.get_affine()

    @staticmethod
    def generate_conversion(orig_file_type, dest_file_type, method):
        conversion = Conversion.ConversionType(orig_file_type, dest_file_type, method)
        _valid = conversion.validate()
        return conversion

    @staticmethod
    def generate_prepper(work_dir, input_file_path, orig_file_type):
        return Preppers.BasePrepper.factory(orig_file_type, work_dir, input_file_path)
        # if orig_file_type in ["dicom", "DICOM"]:
        #     prepworker = Preppers.PrepDicom
        #
        # elif orig_file_type in ["nifti", "NIFTI"]:
        #     prepworker = Preppers.PrepNifti
        #
        # return Preppers.Prepper(
        #     work_dir=work_dir, input_file_path=input_file_path, prepper=prepworker
        # )

    @staticmethod
    def generate_collector(fw_client, orig_dir, file_object, orig_file_type):

        return Collectors.BaseCollector.factory(
            type_=orig_file_type,
            fw_client=fw_client,
            orig_dir=orig_dir,
            file_object=file_object,
        )

    # if file_object.file_type in ["dicom", "DICOM"]:
    #         collworker = Collectors.DicomRoiCollector
    #
    #     elif file_object.file_type in ["nifti", "NIFTI"]:
    #         collworker = Collectors.NiftiRoiCollector
    #
    #     return Collectors.BaseCollector.factory(
    #         fw_client=fw_client,
    #         orig_dir=orig_dir,
    #         file_object=file_object,
    #         collector=collworker,
    #     )

    @staticmethod
    def generate_converter(conversion, orig_dir, roi_dir, combine, bitmask, output_dir):

        return Converters.BaseConverter.factory(
            type_=conversion.method_name,
            orig_dir=orig_dir,
            roi_dir=roi_dir,
            output_dir=output_dir,
            conversion=conversion,
        )

        # if conversion.method_name == "dcm2niix":
        #
        #     if conversion.source == "dicom":
        #         convertworker = Converters.dcm2niix
        #
        # elif conversion.method_name == "slicer-dcmtk":
        #     if conversion.source == "dicom":
        #         convertworker = Converters.slicer_dcmtk
        #
        # elif conversion.method_name == "slicer-gdcm":
        #     if conversion.source == "dicom":
        #         convertworker = Converters.slicer_gdcm
        #
        # elif conversion.method_name == "slicer-arch":
        #     if conversion.source == "dicom":
        #         convertworker = Converters.slicer_arch
        #
        # elif conversion.method_name == "plastimatch":
        #     if conversion.source == "dicom":
        #         convertworker = Converters.plastimatch
        #
        # elif conversion.method_name == 'dicom2nifti':
        #     convertworker = Converters.dicom2nifti
        #

        # converter = Converters.Converter(
        #     orig_dir=orig_dir,
        #     roi_dir=roi_dir,
        #     combine=combine,
        #     bitmask=bitmask,
        #     output_dir=output_dir,
        #     conversion=conversion,
        #     converter=convertworker,
        # )

        return converter

    @staticmethod
    def generate_creator(
        converter, file_object, orig_dir, roi_dir, combine, bitmask, output_dir
    ):

        return Creators.BaseCreator.factory(
            type_=file_object.file_type,
            orig_dir=orig_dir,
            roi_dir=roi_dir,
            output_dir=output_dir,
            base_file_name=file_object.base_name,
            combine=combine,
            bitmask=bitmask,
            converter=converter,
        )

        # if file_object.file_type in ["dicom", "DICOM"]:
        #     createworker = Creators.DicomCreator
        #
        # elif file_object.file_type in ["nifti", "NIFTI"]:
        #     createworker = Creators.NiftiCreator
        #
        # return Creators.Creator(
        #     orig_dir=orig_dir,
        #     roi_dir=roi_dir,
        #     combine=combine,
        #     bitmask=bitmask,
        #     output_dir=output_dir,
        #     base_file_name=file_object.base_name,
        #     creator=createworker,
        #     converter=converter,
        # )

    def process_file(self):
        self.prepper.prep()
        ohifviewer_info = self.collector.collect()
        labels = self.creator.create(ohifviewer_info)
        affine = self.creator.get_affine()
        return ohifviewer_info, labels, affine


"""
Overview:

MeasurementExporter:
    - file_object - holds the file and info on it
    - roi_object - holds the roi and generates the roi.  Will have a "roi_generator" object, which
        can be a "dicom ROI generator", or a "nifti roi generator".  
        For dicoms, there needs to be some mapping between which slice goes to which dicom object.
        This isn't necessary for niftis as it's 1:1.  Therefor, I think the dicom_ROI_Generator should 
        have this property and store it when it gets created.  Later, the ROI_object will be passed into the 
        "ROI exporter", which will know where to look and can handle that.
    - ROI_exporter - probably the workhorse of this class.  This will have to be the one that's unique for 
        each kind of export.  dcm2niix exporter, slicer exporter, plastimatch exporter, dicom2nifti exporter,
        nifti2nifti exporter.
        takes: roi_object, file_object
        returns: nothing but saves the roi file out. 
    - output_type
    

ROI_generator:
    - generates the ROI's from a source file
    will either be dicom generator or nifti generator, each with a "generate()" function.

"""


# Notes:
"""
1. I think self.labels.num_voxels is wrong...wait no JK I think it's correct
2. Working on the collector - it needs 
   a. collect all ROI's related to the given image and
   b. to make the ROI label list
   
   I think this is done well in the original code, but possible room
   for improvement?

3. The collector will need to pass the label/ohif metadata object
back out to the orchestrator I think, because NEXT is the generator,
which will rely on those to move forward.  The generator simply
Generates ROI images in the original image format.
4. Finally the converter.  THE CONVERTER IS A PROPERTY OF THE GENERATOR.
Or maybe it should be generator is part of the converter... But they should be nested
since they need to work together.  If multiple binary labels exist, it will need to:
 a. make the mask iimage for label 1 in the original image format and save
 b. convert that to the desired image format
 c. repeat for all labels.
So this has to be self-contained, it can't loop the right way otherwise.




"""
