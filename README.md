# ROI2nix

Converts ROIs created in the Flwwheel OHIF viewer to NIfTI files.

## Usage Notes

This gear will take as input a NIfTI file segmented with one or more labels in Flywheel's embedded OHIF viewer. It will provide as output NIfTI file representations of these segmentations.  These output NIfTIs can be single files representing each label segmented or combined into one file, depending on configuration (see below). If no ROIs are present, the gear will notify and exit without producing any output. Current maximum number of distinct ROI labels is 63.

### Default Output

* A NIfTI file for each ROI label. These default to binary masks for each region (see below).
* A comma-separated-value (csv) file describing properties of each ROI with respects to ROI label, ROI index, voxel count, and total volume in cubic milimeters. Each line is as below:<br><br>
```<roi_label>, <roi_index>, <roi_voxels>, <roi_volume (mm^3)>```<br><br>
for each ROI. <br>
The ROI volumes are generated by multiplying the total number of voxels by the determinant of the 3x3 submatrix of the affine.

### Optional Output

* The combined ROIs in a bitmasked (sum of powers of two) NIfTI file.
* A 3D Slicer colortable file.

### Inputs

* **Input_File** (required): The NIfTI file with ROIs created in the OHIF viewer

### Parameters

* **save_binary_masks**: Saves individually labeled ROIs as binary masks. Otherwise use bitmasked values (sums of powers of two). Default is True.
* **save_combined_output**: Saves all ROIs in a single bitmasked file. If less than 2 ROIs exists, then this has no effect. Default is False.
* **combined_output_size**: Size of combined bitmasked file: int8|int16|int32|int64. Default is int32.
* **slicer_color_table**: Saves a 3D Slicer colortable file for 3D Slicer integration. Default is False.

### Note

In the combined output file, if any individual ROIs overlap, the voxel value where they overlap will be the sum of the integers representing each ROI.  For instance, if two ROIs are drawn the first will have a value of 1 and the second will have 2 (2^0 and 2^1) so any overlapping voxels will have a value of 3.  This may cause confusion since most viewers will display this in a third color.
