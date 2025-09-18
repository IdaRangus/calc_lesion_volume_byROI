import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd

# -------------- Paths ---------------
lesion_dir = "/Users/idarangus/Desktop/Files/PROJECTS/SA-Axis/lesions" # directory containing stroke lesions 
atlas_path = "/Users/idarangus/Desktop/Files/PROJECTS/SA-Axis/rjhu.nii" # atlas file 
labels_path = "/Users/idarangus/Desktop/Files/PROJECTS/SA-Axis/jhu.txt" # atlas label file 
sa_ranks_path = "/Users/idarangus/Desktop/Files/PROJECTS/SA-Axis/MeanSA_in_JHU_ROIs.csv" # ROI-wise S-A ranks adapted to JHU atlas
output_csv = "/Users/idarangus/Desktop/Files/PROJECTS/SA-Axis/JHU_lesion_volumes_byROI.csv"

# ------------- Load atlas, labels, and S-A ranks ----------------
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata().astype(int)
voxel_vol_cm3 = np.prod(atlas_img.header.get_zooms()) / 1000.0  # mm³ -> cm³

labels = pd.read_csv(labels_path, sep="|", header=None,
                     names=["Index", "Abbrev", "Name", "Type"])

sa_ranks = pd.read_csv(sa_ranks_path)

# -------------- Prepare output dataframe ---------------
results = []

lesion_files = sorted(glob.glob(os.path.join(lesion_dir, "rM*_lesion.nii")))

for lf in lesion_files:
    subj_id = os.path.basename(lf).split("_")[0] 
    lesion_img = nib.load(lf)
    lesion_data = lesion_img.get_fdata() > 0
    
    subj_dict = {"Participant": subj_id}
    
    sa_weighted_sum = 0.0
    lesioned_roi_count = 0
    
    for _, row in labels.iterrows():
        roi_idx = row["Index"]
        roi_abbrev = row["Abbrev"]
        
        mask_roi = atlas_data == roi_idx
        overlap_vox = np.logical_and(mask_roi, lesion_data).sum()
        lesion_vol_cm3 = overlap_vox * voxel_vol_cm3
        
        # get SA rank for this ROI
        sa_val = sa_ranks.loc[sa_ranks['Abbrev'] == roi_abbrev, 'MeanSA']
        sa_val = sa_val.values[0] if not sa_val.empty else np.nan
        
        # save both raw and weighted volumes
        subj_dict[f"{roi_abbrev}_cm3"] = lesion_vol_cm3
        subj_dict[f"{roi_abbrev}_SAweighted"] = lesion_vol_cm3 * sa_val if not np.isnan(sa_val) else np.nan
        
        # update sums for proportional S-A rank
        if lesion_vol_cm3 > 0 and not np.isnan(sa_val):
            sa_weighted_sum += lesion_vol_cm3 * sa_val
            lesioned_roi_count += 1
    
    # compute proportional SA rank
    if lesioned_roi_count > 0:
        subj_dict["Proportional_SA_rank"] = sa_weighted_sum / lesioned_roi_count
    else:
        subj_dict["Proportional_SA_rank"] = np.nan
    
    results.append(subj_dict)

# -------------- Save results ---------------
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved results to {output_csv}")