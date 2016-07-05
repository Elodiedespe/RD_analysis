# import module
import os
import numpy as np
import nibabel as nib
import xml.etree.ElementTree as ET 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil 
from nilearn import plotting, image


def read_roiLabel(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    roi_label=[]

    for label in root.findall('label'):
        rank = label.get('id')
        name = label.get('fullname')
        print rank, name
        roi_label.append(rank)

    roi_label=map(int,roi_label)

    return name, rank, roi_label

def get_roi_mask(roi_filename, label_number):
    mask1 = nib.load(roi_filename)
    roi_affine = mask1.get_affine()
    mask= mask1.get_data()  
    mask = mask.reshape(mask.shape[0:3])

    mask = mask == label_number

    return mask, roi_affine



if __name__ == "__main__":


    # Global parameters
    Path = "/home/edogerde/Bureau/extraction_roi"
    xml_path = "/home/edogerde/Bureau/delineation_space/lpba40.label.xml"
    roi_nii = "/home/edogerde/Bureau/Atlas_plt_roi/label_mni.nii.gz"
    anat_nii = "/home/edogerde/Bureau/Atlas_plt_roi/regis_mni.nii"
    maskfile = "/home/edogerde/Bureau/Atlas_plt_roi"
    when =[163, 164, 101, 61, 46, 62, 47, 68]
    heure = [163, 102, 164, 61, 68, 67, 62, 44, 48]
    Journalier = [161, 162, 47, 45, 101, 102, 163, 48, 62, 166]
    composante2 = [163, 101,102, 61, 67, 68, 62, 48, 47, 46, 43, 161, 89, 90, 165,166]


    mask, affine = get_roi_mask(roi_nii, 1.)
    mask = np.zeros(mask.shape[0:3])
    labels = when
    network_name = 'network'
    for label_number in labels:
        m, _ = get_roi_mask(roi_nii, label_number)
        mask += m

    print mask.shape
    print np.sum(mask)
    elo = nib.Nifti1Image(mask.astype(np.int8), affine)
    nifti_name = os.path.join(maskfile, "%s.nii" % (network_name))
    nib.save(elo, nifti_name)

    plot_name = os.path.join(maskfile, "%s.png" % (network_name))
    plotting.plot_glass_brain(nifti_name, threshold=0.1, display_mode='ortho', cmap="rainbow")
    plt.show()




""" for NewA in NEWA:
A = np.logical_or(A, NewA)"""
