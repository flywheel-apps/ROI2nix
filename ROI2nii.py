from skimage import draw
import numpy as np
import nibabel as nb
import flywheel

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


if __name__ == "__main__":
    fw = flywheel.Client()
    session_id = '5df94ef0f999360025e1b021'
    acquisitions=fw.acquisitions.find(f'parents.session={session_id}')
    acq = fw.get(acquisitions[0].id)
    Nifti = acq.files[1]
    shp = [166, 256, 256]
    Data = np.zeros(shp)

    for ROI in Nifti.info['roi']:
        if ROI["label"] == 'Neck':
            img_path = ROI["imagePath"]
            st = img_path.find('#')
            end = img_path.find(',')
            z_coord = int(img_path[st+3:end])
            print(z_coord)
            #Initialize polygon coordinate lists
            X = []
            Y = []
            if type(ROI["handles"]) == list:
                for h in ROI['handles']:
                    X.append(h['x'])
                    Y.append(h['y'])
            X.append(X[0])
            Y.append(Y[0])
            Data[:,:,z_coord] = poly2mask(X,Y,shp[:2])
    # TODO: Grab affine from original nifti to put here.
    nii = nb.Nifti1Image(Data,np.eye(4))
    nb.save(nii,'Test.nii.gz')

