from dataclasses import dataclass


NIFTI_TYPE = 'nii'
NRRD_TYPE = 'nrrd'



@dataclass
class MethodTypes:
    method: str
    valid_source: list
    valid_dest: list

    def validate(self, source, dest):
        if source.lower() not in self.valid_source:
            raise Exception(f'Invalid source {source} for method {self.method}')
        if dest.lower() not in self.valid_dest:
            raise Exception(f'Invalid destination {source} for method {self.method}')
        return True


dcm2niix_valid_source = ['dicom']
dcm2niix_valid_dest = ['nifti','nrrd']
METHOD_DCM2NIIX = MethodTypes(method='dcm2niix',
                              valid_source=dcm2niix_valid_source,
                              valid_dest=dcm2niix_valid_dest)

slicer_valid_source = ['dicom']
slicer_valid_dest = ['nifti','nrrd']
METHOD_SLICER = MethodTypes(method='slicer',
                              valid_source=slicer_valid_source,
                              valid_dest=slicer_valid_dest)

plastimatch_valid_source = ['dicom', 'nifti']
plastimatch_valid_dest = ['nifti', 'nrrd']
METHOD_PLASTIMATCH = MethodTypes(method='plastimatch',
                              valid_source=plastimatch_valid_source,
                              valid_dest=plastimatch_valid_dest)

dicom2nifti_valid_source = ['dicom']
dicom2nifti_valid_dest = ['nifti']
METHOD_DICOM2NIFTI = MethodTypes(method='dicom2nifti',
                              valid_source=dicom2nifti_valid_source,
                              valid_dest=dicom2nifti_valid_dest)


@dataclass
class ConversionType:
    source: str = ''
    dest: str = ''
    method_name: str = ''
    method: MethodTypes = MethodTypes
    ext: str = ''

    def __post_init__(self):
        if self.method_name in ['dcm2niix']:
            self.method = METHOD_DCM2NIIX
        elif self.method_name in ['slicer-dcmtk', 'slicer-arch', 'slicer-gdcm']:
            self.method = METHOD_SLICER
        elif self.method_name in ['plastimatch']:
            self.method = METHOD_PLASTIMATCH
        elif self.method_name in ['dicom2nifti']:
            self.method = METHOD_DICOM2NIFTI


        lookup = {
                NIFTI_TYPE: ['nifti', 'nii', '.nii'],
                NRRD_TYPE: ['nrrd','.nrrd']
        }

        for e, vals in lookup.items():
            if self.dest.lower() in vals:
                self.ext = e
                break



    def validate(self):
        return self.method.validate(self.source, self.dest)

    def describe(self):
        return f"{self.method.method} {self.source} to {self.dest}"
