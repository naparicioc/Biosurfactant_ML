import os
import statistics
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from io import StringIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def load_fasta(path):

    train_data = {"Name": [],
                 "Fasta": []
                 }
    
    test_data = {"Name": [],
                 "Fasta": []
                 }

    cwd = os.getcwd()
    fasta_file = os.path.join(cwd, "Data", path + ".fasta")

    total_records = 0

    for record in SeqIO.parse(fasta_file, "fasta"):
        
        name = record.id
        fasta = str(record.seq)
        
        total_records += 1

        if len(train_data["Name"]) / total_records >= 0.8:
            test_data["Name"].append(name)
            test_data["Fasta"].append(fasta)

        else:
            train_data["Name"].append(name)
            train_data["Fasta"].append(fasta)
    
    return train_data, test_data
 
def features_fasta(data_dict, type):
     
    features_dict = {"Length": [],
                    "MolWeight": [],
                    "IsoPoint": [],
                    "Aromaticity": [],
                    "InstabilityIndex": [],
                    "Gravy": [],
                    "Flexibility": [],
                    "Helix": [],
                    "Turn": [],
                    "Sheet": [],
                    "Label": [],
                    }
    
    if type == "surfactant":
        label = 1

    else:
        label = 0
     
    failed_indices = []  

    for index, fasta in enumerate(tqdm(data_dict["Fasta"])):

        try:
            protein_analysis = ProteinAnalysis(fasta)
            length = len(fasta)
            molecular_weight = protein_analysis.molecular_weight()
            isoelectric_point = protein_analysis.isoelectric_point()
            aromaticity = protein_analysis.aromaticity()
            instability_index = protein_analysis.instability_index()
            gravy = protein_analysis.gravy()
            flexibility = statistics.mean(protein_analysis.flexibility())
            second_structure = protein_analysis.secondary_structure_fraction()

            features_dict["Length"].append(length)
            features_dict["MolWeight"].append(molecular_weight)
            features_dict["IsoPoint"].append(isoelectric_point)
            features_dict["Aromaticity"].append(aromaticity)
            features_dict["InstabilityIndex"].append(instability_index)
            features_dict["Gravy"].append(gravy)
            features_dict["Flexibility"].append(flexibility)
            features_dict["Helix"].append(second_structure[0])
            features_dict["Turn"].append(second_structure[1])
            features_dict["Sheet"].append(second_structure[2])
            features_dict["Label"].append(label)
        except:
            print(f"No se pudo realizar el estudio de la proteina con el FASTA {fasta}")
            failed_indices.append(index)
    
    for index in failed_indices:
        for key in features_dict:
            features_dict[key].insert(index, 0)

    data_dict.update(features_dict)

    return data_dict

def save_csv(dict_one, dict_two, fold):

    df_one = pd.DataFrame(dict_one)
    df_two = pd.DataFrame(dict_two)

    final_df = pd.concat([df_one, df_two])

    final_df.to_csv("Data/" + fold + ".csv", index = False)


if __name__ == "__main__":

    train_surfactant, test_surfactant = load_fasta("surfactant")
    train_nonsurfactant, test_nonsurfactant = load_fasta("no_surfactant")

    train_surfactant = features_fasta(train_surfactant, "surfactant")
    test_surfactant = features_fasta(test_surfactant, "surfactant")
    train_nonsurfactant = features_fasta(train_nonsurfactant, "no_surfactant")
    test_nonsurfactant = features_fasta(test_nonsurfactant, "no_surfactant")

    save_csv(train_surfactant, train_nonsurfactant, "train")
    save_csv(test_surfactant, test_nonsurfactant, "test")

