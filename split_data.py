import pandas as pd
import os

def split_csv(input_file, output_folder, num_files=4):
    # Read the original CSV file
    df = pd.read_csv(input_file)

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate the number of samples per file
    samples_per_file = len(df) // num_files

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Split the dataframe into num_files parts
    for i in range(num_files):
        start_index = i * samples_per_file
        end_index = (i + 1) * samples_per_file if i < num_files - 1 else len(df)
        output_file = os.path.join(output_folder, f"train_{i + 1}.csv")
        df_part = df.iloc[start_index:end_index]
        df_part.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_csv_file = "Data/train.csv"
    output_folder = "Data/"
    num_files = 4

    split_csv(input_csv_file, output_folder, num_files)