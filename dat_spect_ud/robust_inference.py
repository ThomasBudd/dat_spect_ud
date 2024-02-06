from dat_spect_ud.network_ensemble import network_ensemble
import os
import nibabel as nib
from tqdm import tqdm
from pandas import DataFrame

def dat_spect_robust_inference(path_to_images):
    
    nii_files = [f for f in os.listdir(path_to_images) if f.endswith('.nii.gz') or f.endswith('.nii')]
    
    if len(nii_files) == 0:
        raise FileNotFoundError(f"Found no nifti files at {path_to_images}")
    
    print(f"Using the following niftis: {nii_files}")
    
    # dict for translating 0 and 1 into "normal" and "reduced"
    d = {0: "normal", 1: "reduced"}
    
    acc_ens = network_ensemble('robust')

    pred_list = []
    
    for file in tqdm(nii_files):
        
        im = nib.load(os.path.join(path_to_images, file)).get_fdata()
        acc_vote = d[acc_ens(im)]
        
        pred_list.append(acc_vote)
    
    file_names = [file.split('.')[0] for file in nii_files]
    
    df = DataFrame({'File': file_names,
                    'Predction': pred_list})

    df.to_excel(os.path.join(path_to_images, "robust_classification_results.xlsx"))

    
            
