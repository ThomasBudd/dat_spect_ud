from setuptools import setup
import os

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='dat_spect_ud',
    url='https://https://github.com/ThomasBudd/dat_spect_ud',
    author='Thomas Buddenkotte',
    author_email='thomasbuddenkotte@googlemail.com',
    # Needed to actually package something
    packages=['dat_spect_ud'],
    # Needed for dependencies
    install_requires=[
            "torch>=1.7.0",
            "tqdm",
            "nibabel",
            "pandas"
      ],
    # *strongly* suggested for sharing
    version='1.0',
    description='inference code for uncertainty detection on DAT SPECT images',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
