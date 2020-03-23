# ROI2nix

Converts ROIs created in the Flwwheel OHIF viewer to NIfTI files.

## Usage Notes

This gear will take as input a NIfTI file segmented with one or more labels in Flywheel's embedded OHIF viewer. It will provide as output NIfTI file representations of these segmentations.  These output NIfTIs can be single files representing each label segmented or combined into one file, depending on configuration (see below). If no ROIs are present, the gear will notify and exit without producing any output. Current maximum number of distinct ROI labels is 63.

### Default Output

* A NIfTI file for each ROI label. These default to binary masks for each region (see below).
* A comma-separated-value (csv) file describing properties of each ROI with respects to ROI label, ROI index, voxel count, and total volume in cubic milimeters. Each line is as below:<br>
`<roi_label>, <roi_index>, <roi_voxels>, <roi_volume (mm^3)>`<br>
for each ROI. <br>
The ROI volumes are generated by multiplying the total number of voxels by the determinant of the 3x3 submatrix of the affine.

### Optional Output

* The combined ROIs in a bitmasked (sum of powers of two) NIfTI file.
* A 3D Slicer colortable file.

### inputs

* Input_File (required): The NIfTI file with ROIs created in the OHIF viewer

### parameters

* **save_binary_masks**: Saves individually labeled ROIs as binary masks. Otherwise use bitmasked values (sums of powers of two). Default is True.
* **save_combined_output**: Saves all ROIs in a single bitmasked file. If less than 2 ROIs exists, then this has no effect. Default is False.
* **slicer_color_table**: Saves a 3D Slicer colortable file for integration. Default is False.
