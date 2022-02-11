from dataclasses import dataclass


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
method_dcm2niix = MethodTypes(method='dcm2niix',
                              valid_source=dcm2niix_valid_source,
                              valid_dest=dcm2niix_valid_dest)

slicer_valid_source = ['dicom']
slicer_valid_dest = ['nifti','nrrd']
method_slicer = MethodTypes(method='slicer',
                              valid_source=dcm2niix_valid_source,
                              valid_dest=dcm2niix_valid_dest)


@dataclass
class ConversionType:
    source: str
    dest: str
    method_name: str
    method: MethodTypes = MethodTypes

    def __post_init__(self):
        if self.method_name=='dcm2niix':
            self.method = method_dcm2niix
        elif self.method_name=='slicer':
            self.method = method_slicer

    def validate(self):
        return self.method.validate(self.source, self.dest)

    def describe(self):
        return f"{self.method.method} {self.source} to {self.dest}"
