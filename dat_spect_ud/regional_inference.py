from dat_spect_ud.network_ensemble import regional_ensemble
import os
import nibabel as nib
from tqdm import tqdm
from pandas import DataFrame

def dat_spect_regional_inference(path_to_images):
    
    nii_files = [f for f in os.listdir(path_to_images) if f.endswith('.nii.gz') or f.endswith('.nii')]
    
    if len(nii_files) == 0:
        raise FileNotFoundError(f"Found no nifti files at {path_to_images}")
    
    print(f"Using the following niftis: {nii_files}")
    
    # dict for translating 0 and 1 into "normal" and "reduced"
    regions = ['cau_r', 'cau_l', 'put_r', 'put_l']
    categ_to_str = {0: 'normal',
                    1: 'boarderline reduced', 
                    2: 'moderatly reduced',
                    3: 'strongly reduced', 
                    4: 'almost missing'} 
    
    reg_ens = regional_ensemble()

    reg_pred_dict = {region: [] for region in regions}
    
    for file in tqdm(nii_files):
        
        im = nib.load(os.path.join(path_to_images, file)).get_fdata()
        pred = reg_ens(im)
        
        for region, p in zip(regions, pred):
            reg_pred_dict[region].append(categ_to_str[p])
        
    file_names = [file.split('.')[0] for file in nii_files]
    
    df = DataFrame({'File': file_names,
                    **reg_pred_dict})

    df.to_excel(os.path.join(path_to_images, "regional_classification_results.xlsx"))

    
            
