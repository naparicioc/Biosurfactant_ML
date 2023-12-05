import os
import subprocess
import statistics
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from io import StringIO
from Bio.Seq import Seq
from Bio.SeqUtils import seq3
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

    print("--------------------Leyendo el archivo .fasta--------------------")
    for record in tqdm(SeqIO.parse(fasta_file, "fasta")):
        
        name = record.name.split("|")[1]
        fasta = str(record.seq)
        
        total_records += 1

        if len(train_data["Name"]) / total_records >= 0.8:
            test_data["Name"].append(name)
            test_data["Fasta"].append(fasta)

        else:
            train_data["Name"].append(name)
            train_data["Fasta"].append(fasta)
    
    train_data["Name"].append("Mutante6")
    train_data["Name"].append("Mutante12")
    train_data["Fasta"].append("QETAMTMITPSSELTLTKGTSPAGLNEFALVSGQFHTSRVPCPRANSRPLNSIRPIVSRITIHWPSFY")
    train_data["Fasta"].append("LVKRRPVNCNTTHYRANIRPRIRPWAWHSILCEIVIRSQGRIRLNLQDSLGLILSLASWSLLP")
    
    return train_data, test_data

def calcular_descriptores(secuencia):
    sec_biopython = Seq(secuencia)
    
    contenido_aminoacidos = {}

    # Lista predefinida de códigos de una letra
    codigo_una_letra = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    for aminoacido in codigo_una_letra:
        contenido_aminoacidos[aminoacido] = sec_biopython.count(aminoacido)

    list_values = list(contenido_aminoacidos.values())

    return list_values
 
def features_fasta(data_dict, protein_type):
    features_dict = {
        "Length": [],
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
        # Additional molecular descriptors
        "ChargeAtPH7": [],
        "AliphaticIndex": [],
        "FragilityIndex": [],
        # iFeatureOmegaCLI features
        'A': [], 
        'C': [], 
        'D': [], 
        'E': [], 
        'F': [], 
        'G': [], 
        'H': [], 
        'I': [], 
        'K': [], 
        'L': [], 
        'M': [], 
        'N': [], 
        'P': [], 
        'Q': [], 
        'R': [], 
        'S': [], 
        'T': [], 
        'V': [], 
        'W': [], 
        'Y': [],
        "TensionInterfacial": []
    }

    if protein_type == "surfactant":
        label = 1
    else:
        label = 0

    failed_indices = []

    for index, fasta in enumerate(tqdm(data_dict["Fasta"])):
        try:
            if "X" in fasta:
                fasta = fasta.replace("X", "")

            if "U" in fasta:
                fasta = fasta.replace("U", "")

            protein_analysis = ProteinAnalysis(fasta)
            length = len(fasta)
            molecular_weight = protein_analysis.molecular_weight()
            isoelectric_point = protein_analysis.isoelectric_point()
            aromaticity = protein_analysis.aromaticity()
            instability_index = protein_analysis.instability_index()
            gravy = protein_analysis.gravy()
            flexibility = statistics.mean(protein_analysis.flexibility())
            second_structure = protein_analysis.secondary_structure_fraction()
            charge_at_ph7 = protein_analysis.charge_at_pH(7)
            alanine_percentage = protein_analysis.get_amino_acids_percent()['A']
            isoleucine_percentage = protein_analysis.get_amino_acids_percent()['I']
            aliphatic_index = (alanine_percentage + 2 * isoleucine_percentage) / 3
            fragility_index = aliphatic_index / instability_index

            # Calcular descriptores de iFeatureOmega
            amino_values = calcular_descriptores(fasta)
            
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
            features_dict["ChargeAtPH7"].append(charge_at_ph7)
            features_dict["AliphaticIndex"].append(aliphatic_index)
            features_dict["FragilityIndex"].append(fragility_index)
            features_dict["A"].append(amino_values[0])  
            features_dict["C"].append(amino_values[1])
            features_dict["D"].append(amino_values[2])
            features_dict["E"].append(amino_values[3])
            features_dict["F"].append(amino_values[4])
            features_dict["G"].append(amino_values[5])
            features_dict["H"].append(amino_values[6])
            features_dict["I"].append(amino_values[7])
            features_dict["K"].append(amino_values[8])
            features_dict["L"].append(amino_values[9])
            features_dict["M"].append(amino_values[10])
            features_dict["N"].append(amino_values[11])
            features_dict["P"].append(amino_values[12])
            features_dict["Q"].append(amino_values[13])
            features_dict["R"].append(amino_values[14])
            features_dict["S"].append(amino_values[15])
            features_dict["T"].append(amino_values[16])
            features_dict["V"].append(amino_values[17])
            features_dict["W"].append(amino_values[18])
            features_dict["Y"].append(amino_values[19])
            features_dict["TensionInterfacial"].append(0)

        except Exception as e:
            print(f"Failed to analyze the protein with FASTA {fasta}. Error: {e}")
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

    train_surfactant, test_surfactant = load_fasta("biosurfactant_sequences")
    train_nonsurfactant, test_nonsurfactant = load_fasta("nonbiosurfactant_sequences")

    train_surfactant = features_fasta(train_surfactant, "surfactant")
    test_surfactant = features_fasta(test_surfactant, "surfactant")

    train_nonsurfactant = features_fasta(train_nonsurfactant, "no_surfactant")
    test_nonsurfactant = features_fasta(test_nonsurfactant, "no_surfactant")

    save_csv(train_surfactant, train_nonsurfactant, "train")
    save_csv(test_surfactant, test_nonsurfactant, "test")

