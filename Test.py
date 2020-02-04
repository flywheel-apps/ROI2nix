from run import *
import nibabel as nib

if __name__ == "__main__":
    fw = flywheel.Client()
    session_id = '5df94ef0f999360025e1b021'
    session_id = '5e389eaf046aa606d91973bb'
    acquisitions = fw.acquisitions.find(f'parents.session={session_id}')
    acq = fw.get(acquisitions[0].id)
    file_obj = acq.files[0]


    # dictionary for labels, index, R, G, B, A
    labels = OrderedDict()

    # only doing this for toolType=freehand
    # TODO: Consider other closed regions:
    # rectangleRoi, ellipticalRoi
    if 'roi' in file_obj.info.keys():
        for roi in file_obj.info['roi']:
            if (roi['toolType'] == 'freehand') and \
                    (roi['label'] not in labels.keys()):
                # Only if annotation type is a polygon, then grab the
                # label, create a 2^x index for bitmasking, grab the color
                # hash (e.g. #fbbc05), and translate it into RGB
                labels[roi['label']] = {
                    'index': int(2**(len(labels))),
                    'color': roi['color'],
                    'RGB': [
                        int(roi['color'][i: i+2], 16)
                        for i in [1, 3, 5]
                    ]
                }
    #shp = [166, 256, 256]
    shp = [208, 300, 320]
    data = np.zeros(shp, dtype=int)
    for label in labels:
        indx = labels[label]['index']
        data += indx * label2data(label, shp, file_obj.info)

    nib.save(nib.Nifti1Pair(data.astype(float), np.eye(4)), 'test.nii.gz')

    for label in labels:
        indx = labels[label]['index']
        label_nii = nb.Nifti1Pair(
            np.bitwise_and(data, indx).astype(float),
            np.eye(4)
        )

        nb.save(
            label_nii,
            label + '_label_.nii.gz'
        )
