from run import *
import nibabel as nib

if __name__ == "__main__":
    fw = flywheel.Client()
    session_id = '5df94ef0f999360025e1b021'
    acquisitions=fw.acquisitions.find(f'parents.session={session_id}')
    acq = fw.get(acquisitions[0].id)
    Nifti = acq.files[1]
    shp = [166, 256, 256]
    data = np.zeros(shp)
    data += label2data('Brain', shp, Nifti.info)
    print(data.shape)
    
    nib.save(nib.Nifti1Pair(data,np.eye(4)),'test.nii.gz')