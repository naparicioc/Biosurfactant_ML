import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import pubchempy as pcp
from rdkit.Chem import Descriptors

def read_csv(path):

    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, "Data", path + ".csv"))

    return data

def get_smiles(data):

    list_smiles = []

    for i in tqdm(range(0, len(data))):
        
        id = int(data.iloc[i]["id"])

        try: 
            compound = pcp.Compound.from_cid(id)
            smiles = compound.canonical_smiles
            list_smiles.append(smiles)
        
        except:
            print(f"No se ha podido determinar el SMILES del ID {id}")

    data["smiles"] = list_smiles

    return data

def smiles_features(data):

    mw = []
    logp = []
    hbd = []
    hba = []
    numCarbonAtoms = []
    numAromaticRings = []
    meltingPoint = []
    num_atoms = []
    connectivity_index = []
    valency = []

    for smiles in tqdm(data["smiles"]):
    
        try:
            mol = Chem.MolFromSmiles(smiles)
            mw.append(Descriptors.MolWt(mol))
            logp.append(Descriptors.MolLogP(mol))
            hbd.append(Descriptors.NumHDonors(mol))
            hba.append(Descriptors.NumHAcceptors(mol))
            numCarbonAtoms.append(Descriptors.HeavyAtomCount(mol))
            numAromaticRings.append(Descriptors.NumAromaticRings(mol))
            meltingPoint.append(Descriptors.MolMR(mol))
            num_atoms.append(Descriptors.HeavyAtomCount(mol))
            connectivity_index.append(Descriptors.FractionCSP3(mol))
            valency.append(Descriptors.NumValenceElectrons(mol))

        except:
            print(f"El descriptor no pudo ser determinado para la molécula con SMILES {smiles}")

    data["ExactMolWt"] = mw
    data["LogP"] = logp
    data["NumHDonors"] = hbd
    data["NumHAcceptors"] = hba
    data["CarbonosNum"] = numCarbonAtoms
    data["AromaticRingsNum"] = numAromaticRings
    data["MeltingPoint"] = meltingPoint
    data["NumAtoms"] = num_atoms
    data["Connectivity"] = connectivity_index
    data["Valency"] = valency

    return data

def save_csv(data, file_name):

    cwd = os.getcwd()
    data.to_csv(os.path.join(cwd, "Data", file_name + ".csv"), index=False)

if __name__ == "__main__":

    print("Se van a cargar los .csv con los IDs de las moléculas")
    train = read_csv("id_train")
    test = read_csv("id_test")

    train = get_smiles(train)
    test = get_smiles(test)
    print("Se obtuvieron los SMILES de cada conjunto de datos")

    train = smiles_features(train)
    test = smiles_features(test)
    print("Se determinaron los descriptores molecules de la base de datos")

    save_csv(train, "train")
    save_csv(test, "test")


