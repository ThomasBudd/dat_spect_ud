Inference code for DAT SPECT image analysis using deep learning.

The code covers the inferences of the models used in our publications (see below).

For questions please contact thomasbuddenkotte@googlemail.com or r.buchert@uke.de

# Publications:

1. Budenkotte, T., Apostolova, I., Opfer, R., Krüger, J., Klutmann, S., & Buchert, R. (2023). Automated identification of uncertain cases in deep learning-based classification of dopamine transporter SPECT to improve clinical utility and acceptance. European journal of nuclear medicine and molecular imaging, 10.1007/s00259-023-06566-w. Advance online publication. https://doi.org/10.1007/s00259-023-06566-w
1. Buddenkotte, T., Buchert, R. (2024) "Unrealistic data augmentation improves the robustness of deep learning-based classification of dopamine transporter SPECT against between-sites and between-cameras variability" (currently under revision)

# Installation

To install the library simply clone this repository and install with pip

```
git clone https://github.com/ThomasBudd/dat_spect_ud
cd dat_spect_ud
pip install -e .
```

# Running inference

The library expects the images to be preprocessed as described in the papers, i.e. that the registration to the MNI space was performed, the voxel size was interpolated to 2mm isotropic voxels and that the 2d slabs were computed. The resulting slabs must be saved as nifti images. The code expectes the path to the folder containing the images as input and will write the results in an .xlsx file into it.

## Uncertainty detection
To run the uncerintaty detection models presented in [1] use
```
from dat_spect_ud.ud_inference import dat_spect_ud_inference
dat_spect_ud_inference(PATH_TO_IMAGES_FOLDER)
```
The image will be classified according to the categories "normal", "reduced", and "uncertain".

## Robust classification

To run the robust models trained with the novel data augmentation presented in [2] use
```
from dat_spect_ud.robust_inference import dat_spect_robust_inference
dat_spect_robust_inference(PATH_TO_IMAGES_FOLDER)
```
The image will be classified according to the categories "normal" and "reduced".
