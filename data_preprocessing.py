import pandas as pd
import os
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


def cleandb(db):
    # Fill in null values for given columns
    columns = ['DVP', 'VCS', 'histologie']
    default_val = [0, 0, 1]
    for idx, c in enumerate(columns):
        db.loc[:, c] = db.loc[:, c].fillna(default_val[idx])

    # Get rid of rows with null values for given columns
    columns = ['chirurgie']
    for c in columns:
        db = db[db[c].notnull()]

    # Transform dates. Make sure manually that date fields conform to pattern
    date_fields = ['chirurgie', 'bilan', 'naissance']
    date_pattern = "%d/%m/%Y"
    for d in date_fields:
        db.loc[:, d] = pd.to_datetime(db[d], format=date_pattern)

    # Normalize dates
    date_fields = ['chirurgie', 'bilan']
    reference_date = 'naissance'
    for d in date_fields:
        db.loc[:, 'age_at_' + d] = (db[d] - db[reference_date]) \
                                    .apply(lambda x: x.astype('timedelta64[D]').astype('int'))
    db.loc[:, 'chirurgie_delay_' + d] = (db['bilan'] - db['chirurgie']) \
                                         .apply(lambda x: x.astype('timedelta64[D]').astype('int'))

    # Transform categories to integer values
    columns = ['radiotherapie', 'chimiotherapie']
    cat_values = [[0, 1], [0, 1]]
    for idx, c in enumerate(columns):
        db.loc[:, c] = (db[c].astype('category').cat
                        .rename_categories(cat_values[idx]).astype('int'))

    return db


def create_score_db(score, db, orig_col):
    col = ['delta_' + score, 'age_at_bilan', 'age_diff_bilan'] + orig_col
    sdb = pd.DataFrame(columns=col)
    db = db[db[score].notnull()]
    db.loc[:, score] = db[score].astype('int')
    col_counter = 0
    last_patient = ''
    last_row = ''
    patients = list(set(db['patient'].tolist()))
    for patient in patients:
        pdb = db[db['patient'] == patient]
        pdb = pdb.sort(columns='bilan')
        # print pdb
        ind = pdb.index.tolist()
        if len(ind) == 1:
            continue
        # Extract score difference for last score - first score.
        last_row = pdb.loc[ind[0]]
        row = pdb.loc[ind[-1]]
        assert((row['bilan'] - last_row['bilan']).value > 0)
        delta = row[score] - last_row[score]
        b1 = last_row['age_at_bilan']
        b2 = row['age_at_bilan']
        bdiff = b2 - b1
        other_values = row[orig_col].values.tolist()
        sdb.loc[col_counter] = [delta, b2, bdiff] + other_values
        col_counter += 1
        last_row = row

        # # Creates all differences between scores
        # for idx, row in pdb.iterrows():
        #     if row['patient'] != last_patient:
        #         last_row = row
        #         last_patient = row['patient']
        #         continue
        #     # print row['patient'], (row['bilan'] - last_row['bilan']).value
        #     assert((row['bilan'] - last_row['bilan']).value > 0)
        #     delta = row[score] - last_row[score]
        #     b1 = last_row['age_at_bilan']
        #     b2 = row['age_at_bilan']
        #     bdiff = b2 - b1
        #     other_values = row[orig_col].values.tolist()
        #     sdb.loc[col_counter] = [delta, b2, bdiff] + other_values
        #     col_counter += 1
        #     last_row = row
    return sdb


def extract_brain_rad(db, rad_column, rad_dir, stat, include_chim=False):
    """Replaces radiation presence by stat on whole brain ROI.

    Assumes brain mask and radiation nifti file is in rad_dir."""
    brain_mask_file = 'BrainMask_to_rd.nii.gz'
    extracted_rad_stat = {}  # Memoization of radiation statistic
    for idx, row in db.iterrows():
        if row[rad_column] == 1:
            sub_id = row['patient']
            if sub_id in extracted_rad_stat:
                db.loc[idx, rad_column] = extracted_rad_stat[sub_id]
            else:
                mask_path = os.path.join(rad_dir, sub_id, brain_mask_file)
                mask_check = os.path.isfile(mask_path)
                rad_path = os.path.join(rad_dir, sub_id, sub_id + '.nii')
                rad_check = os.path.isfile(rad_path)
                if mask_check and rad_check:
                    masker = NiftiMasker(mask_path)
                    rad_stat = stat(masker.fit_transform(rad_path))
                    extracted_rad_stat[sub_id] = rad_stat
                    db.loc[idx, rad_column] = rad_stat
                else:
                    db.loc[idx, rad_column] = None
        elif not include_chim:
            db.loc[idx, rad_column] = None

    db = db[db[rad_column].notnull()]
    return db


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


def extract_atlas_rad(db, rad_column, rad_dir, stat, pca_explained_var=None,
                      include_chim=False):
    """Replaces radiation presence by stat on ROIs from atlas.

    Assumes brain mask and radiation nifti file is in rad_dir."""
    atlas_mask_file = 'labels_to_rd.nii.gz'

    xml_path = os.path.join(rad_dir, 'lpba40.label.xml')
    labels, rois_name = read_roiLabel(xml_path)
    # labels = labels[:2]  # for debugging
    # rois_name = rois_name[:2]  # for debugging
    labels.append(1000)
    rois_name.append('rest_brain')

    brain_mask_file = 'BrainMask_to_rd.nii.gz'
    extracted_rad_stat = {}  # Memoization of radiation statistic

    # PCA estimation
    for idx, row in db.iterrows():
        if row[rad_column] == 1:
            sub_id = row['patient']
            if sub_id in extracted_rad_stat:
                continue
            else:
                atlas_mask_path = os.path.join(rad_dir, sub_id,
                                               atlas_mask_file)
                atlas_mask_check = os.path.isfile(atlas_mask_path)
                brain_mask_path = os.path.join(rad_dir, sub_id,
                                               brain_mask_file)
                brain_mask_check = os.path.isfile(brain_mask_path)
                rad_path = os.path.join(rad_dir, sub_id, sub_id + '.nii')
                rad_check = os.path.isfile(rad_path)
                if atlas_mask_check and brain_mask_check and rad_check:
                    extracted_rad = []

                    brain_masker = NiftiMasker(brain_mask_path)
                    atlas_brain = brain_masker.fit_transform(atlas_mask_path)
                    atlas_brain = atlas_brain.astype(np.int16)
                    atlas_brain[atlas_brain == 0] = 1000
                    atlas_brain = brain_masker.inverse_transform(atlas_brain)

                    for idx, label in enumerate(labels):
                        print ('processing ROI ' + str(label) + ' ' +
                               rois_name[idx] + ' for subject ' + sub_id)
                        masker = NiftiMasker(get_roi_mask(atlas_brain,
                                             label))
                        rad_stat = stat(masker.fit_transform(rad_path))
                        extracted_rad.append(rad_stat)

                    extracted_rad_stat[sub_id] = extracted_rad

    rad_data = np.vstack([x for x in extracted_rad_stat.values()])
    scaler = StandardScaler()
    rad_data = scaler.fit_transform(rad_data)
    if pca_explained_var is not None:
        pca = PCA(pca_explained_var, whiten=True)
    else:
        pca = PCA(whiten=True)
    pca.fit(rad_data)
    components_name = ['component_{0:02d}'.format(x)
                       for x in range(1, pca.n_components_ + 1)]

    # Modify db to include pca components
    for c in components_name:
        db[c] = ''
    for idx, row in db.iterrows():
        if row[rad_column] == 1:
            sub_id = row['patient']
            atlas_mask_path = os.path.join(rad_dir, sub_id,
                                           atlas_mask_file)
            atlas_mask_check = os.path.isfile(atlas_mask_path)
            brain_mask_path = os.path.join(rad_dir, sub_id,
                                           brain_mask_file)
            brain_mask_check = os.path.isfile(brain_mask_path)
            rad_path = os.path.join(rad_dir, sub_id, sub_id + '.nii')
            rad_check = os.path.isfile(rad_path)
            if atlas_mask_check and brain_mask_check and rad_check:
                rois_v = np.array(extracted_rad_stat[sub_id]).reshape(1, -1)
                components_value = pca.transform(scaler.transform(rois_v))
                components_value = np.ravel(components_value)
                for cidx, c in enumerate(components_name):
                    db.loc[idx, c] = components_value[cidx]
            else:
                db.loc[idx, rad_column] = None
        elif not include_chim:
            db.loc[idx, rad_column] = None
        else:
            for c in components_name:
                db.loc[idx, c] = 0

    db = db[db[rad_column].notnull()]
    db = db.drop(rad_column, 1)
    return db, rois_name, pca, components_name, rad_data


def generate_preprocessed_tables(db_name, db_path, scores, desired_columns,
                                 rad_dir, rad_column='radiotherapie',
                                 rad_type='presence', stat=np.sum,
                                 pca_explained_var=0.95, include_chim=False,
                                 calc_scores=False):
    db = pd.read_csv(os.path.join(db_path, db_name + '.csv'))
    db = cleandb(db)
    if rad_type == 'brain':
        db = extract_brain_rad(db, rad_column, rad_dir, stat, include_chim)
        desired_columns = [rad_column] + desired_columns
    elif rad_type == 'atlas':
        db, rois_name, pca, comp_col, rd = extract_atlas_rad(db, rad_column,
                                                         rad_dir, stat,
                                                         pca_explained_var,
                                                         include_chim)
        desired_columns = comp_col + desired_columns
    else:
        desired_columns = [rad_column] + desired_columns

    if rad_type == 'presence' and not include_chim:
        for idx, row in db.iterrows():
            if row[rad_column] == 0:
                db.loc[idx, rad_column] = None
        db = db[db[rad_column].notnull()]

    tables = []
    if calc_scores:
        for score in scores:
            sdb = create_score_db(score, db, desired_columns)
            tables.append(sdb)
    else:
        tables.append(db)

    if rad_type == 'atlas':
        return tables, pca, rd
    else:
        return tables, None, None


if __name__ == "__main__":
    # Example generating and saving preprocessed tables
    # db_name = 'all_patients'
    # db_path = os.path.join('behav')
    db_name = 'hippomuse_elotin'
    db_path = os.path.join('behav_hippomuse')
    # scores = ['QIT', 'QIV', 'QIP', 'ICV', 'IRP', 'IMT', 'IVT', 'Vocabulaire',
    #           'Similitudes', 'Cubes', 'score_raisonnement', 'score_verbale']
    # desired_columns = ['patient', 'chimiotherapie', 'histologie', 'DVP',
    #                     'VCS','age_at_chirurgie', 'chirurgie_delay_bilan']
    scores = ['hippo']
    desired_columns = ['patient', 'chimiotherapie', 'histologie', 'DVP', 'VCS',
                       'age_at_chirurgie', 'chirurgie_delay_bilan', 'music',
                       'reconnaissanceImageP1', 'reconnaissanceOdeurP1',
                       'musique_P1', 'musique_P2', 'when_P1', 'heure_P1',
                       'repereJournalier_P1', 'what_P1', 'episodicite_P1',
                       'when_P2', 'heure_P2', 'repereJournalier_P2', 'what_P2',
                       'episodicite_P2', 'delta_when_P2', 'delta_heure_P2',
                       'delta_repereJournalier_P2', 'delta_what_P2',
                       'delta_episodicite_P2', 'groupe']

    rad_column = 'radiotherapie'
    rad_dir = os.path.join('extraction_roi')
    rad_type = 'atlas'
    stat = np.sum
    stat_str = 'sum'
    pca_explained_var = 0.95
    include_chim = True
    stat_list = [(np.sum, 'sum'), (np.mean, 'mean'), (np.max, 'max'),
                 (np.min, 'min'), (np.median, 'median'),
                 (np.var, 'var')]

    for stat, stat_str in stat_list:
        for rad_type in ['presence', 'brain', 'atlas']:
            tables, pca, rd = generate_preprocessed_tables(db_name, db_path, scores,
                                                       desired_columns, rad_dir,
                                                       rad_column, rad_type, stat,
                                                       pca_explained_var, include_chim)
            save_to = os.path.join(db_path, 'preprocessed_hippo_RD')
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            if rad_type == 'atlas':
                save_name = (db_name + '_PCA_' + str(stat_str) + '.pkl')
                joblib.dump(pca, os.path.join(save_to, save_name))
                save_name = (db_name + '_RD_' + str(stat_str) + '.pkl')
                joblib.dump(rd, os.path.join(save_to, save_name))
            for idx, table in enumerate(tables):
                if rad_type == 'presence':
                    save_name = db_name + '_' + scores[idx] + '_' + rad_type + '_M.csv'
                else:
                    save_name = (db_name + '_' + scores[idx] + '_' + str(stat_str) +
                                 '_' + rad_type + '_M.csv')
                table.to_csv(os.path.join(save_to, save_name), index=None)


    # save_to = os.path.join(db_path, 'preprocessed')
    # save_name = (db_name + '_PCA_' + str(stat_str) + '.pkl')
    # pca = joblib.load(os.path.join(save_to, save_name))
    #roi_betas = pca.inverse_transform(component_betas)
    #save_to = os.path.join("/neurospin/grip/protocols/MRI/dosimetry_elodie_2015/clemence/ANALYSIS", 'preprocessed')
    #save_name = ("all_patients_PCA_max.pkl")
    #pca = joblib.load(os.path.join(save_to, save_name))
    #roi_betas = pca.inverse_transform(-0.10130023 -1.19442987 -1.14532634 -1.94837162)
