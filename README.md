Inference code for DAT SPECT image analysis with uncertainty detection.
For questions please contact thomasbuddenkotte@googlemail.com or r.buchert@uke.de

Publication: "Automated identification of uncertain cases in deep learning-based classification of dopamine transporter SPECT to improve clinical utility and acceptance"

# Installation

To install the library simply clone this repository and install with pip

```
git clone https://github.com/ThomasBudd/dat_spect_ud
cd dat_spect_ud
pip install -e .
```

# Running inference

The library expects the images to be preprocessed as described in the paper, i.e. that the registration to the MNI space was performed, the voxel size was interpolated to 2mm isotropic voxels and that the 2d slabs were computed. The resulting slabs must be saved as nifti images. The code expectes the path to the folder containing the images as input and will write the results in an .xlsx file into it. The python syntax for this is the following:


```
from dat_spect_ud.ud_inference import dat_spect_ud_inference
dat_spect_ud_inference(PATH_TO_IMAGES_FOLDER)
```
