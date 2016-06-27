import numpy as np
from scipy import ndimage
import os
import nibabel as nib
import xml.etree.ElementTree as ET
from sklearn.externals import joblib


def get_roi_center(data):
    """Get ROI center of mass.
    Get back coordinate in img space and in coordinate space.
    Also actual center of mass.
    """
    center_coords = ndimage.center_of_mass(np.abs(data))
    x_map, y_map, z_map = center_coords[:3]
    voxel = [round(x) for x in center_coords]

    return voxel[:3]


def read_roiLabel(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    names = []

    for field in root.findall('label'):
        label = field.get('id')
        if int(label) != 0:
            name = field.get('fullname')
            # print label, name
            labels.append(int(label))
            names.append(str(name))

    return labels, names


def get_roi_mask(roi_filename, label_number=1):
    if type(roi_filename) == nib.Nifti1Image:
        img = roi_filename
    else:
        img = nib.load(roi_filename)
    mask = img.get_data()
    mask = mask.reshape(mask.shape[0:3])
    mask = mask == label_number
    return nib.Nifti1Image(mask.astype(np.int8), img.get_affine())


if __name__ == "__main__":
    rad_dir = os.path.join('extraction_roi')
    atlas_path = 'ANTS9-5Years3T_label_regis_head.nii.gz'
    xml_path = os.path.join(rad_dir, 'lpba40.label.xml')
    labels, rois_name = read_roiLabel(xml_path)
    coords = []
    for label in labels:
        coords.append(get_roi_center(get_roi_mask(atlas_path,
                                                  label).get_data()))
    order = [np.array(c).argsort() for c in zip(*coords)]
    coord_value = [np.array(c)[order[ci]] for ci, c in enumerate(zip(*coords))]
    joblib.dump(order, 'behav_hippomuse/coord_order.pkl')
    joblib.dump(coord_value, 'behav_hippomuse/coord_value.pkl')
