from shutil import copyfile
import pandas as pd
import numpy as np
import glob
from pymatgen import Structure
from matminer.featurizers.structure import CGCNNFeaturizer

properties = pd.read_csv("C:/Users/sterg/Documents/GitHub/cgcnn/data/cif-K_VRH/id_prop.csv",header=None)
properties.shape

structures = []
#for structure_file in os.listdir("C:/Users/sterg/Documents/GitHub/cgcnn/data/cif-K_VRH/*.cif"):
for structure_file in glob.glob("C:/Users/sterg/Documents/GitHub/cgcnn/data/cif-K_VRH/*.cif"):
    #structure_path = 'C:/Users/sterg/Documents/GitHub/cgcnn/data/cif-only-K_VRH/'+structure_file
    structure_path = structure_file
    structure = Structure.from_file(structure_path)
    structures.append(structure)
df = pd.DataFrame({"K_VRH": properties[1], "structure": structures})
print(df) # make sure the dataframe appears like you intended
df.to_pickle("C:/Users/sterg/Documents/GitHub/cgcnn/data/cif-K_VRH.p")

#%%
featurizer = CGCNNFeaturizer(task='regression', atom_init_fea=None, pretrained_name='bulk-moduli', warm_start_file='C:\\Users\\sterg\\Documents\\GitHub\\cgcnn\\data\\checkpoint.pth.tar', warm_start_latest=False, save_model_to_dir=None, save_checkpoint_to_dir=None, checkpoint_interval=100, del_checkpoint=True)

#%%
featurizer.fit(df.structure,df.K_VRH)

#%%
df.structure[0]

features = featurizer.featurize_many(df.structure,ignore_errors=True,return_errors=False,pbar=True)

X=np.array(features)
print(X.shape)
X[2500]

savepath = "../../../cgcnn/data/K_VRH-features.csv"
copypath = "../../../phylo-mat/code/data/K_VRH-features.csv"
np.savetxt(savepath, X, delimiter=",")
copyfile(savepath,copypath)

X.shape