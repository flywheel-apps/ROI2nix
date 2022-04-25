from dataclasses import dataclass
from flywheel import FileEntry

@dataclass
class FileObject:
    input_file_path: str
    flywheel_file: FileEntry
    file_type: str = None

    def __post_init__(self):
        # Removes the need to reload full parents when getting files
        self.flywheel_file = self.flywheel_file.reload()
        self.file_type = self.flywheel_file.type
        self.base_name = self.flywheel_file.name


