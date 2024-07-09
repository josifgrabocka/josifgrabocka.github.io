
import pandas as pd
import re

def process_log_file(input_file_path, output_file_path):
    # Step 1: Load the file content
    with open(input_file_path, 'r') as file:
        content = file.readlines()

    # Step 2: Remove all rows that do not start with 'Dataset' or 'Epoch'
    filtered_content = [line for line in content if line.startswith('Dataset') or line.startswith('Epoch')]

    # Step 3: Merge every odd and even row
    merged_content = []
    for i in range(0, len(filtered_content), 2):
        if i + 1 < len(filtered_content):
            merged_line = filtered_content[i].strip() + ", " + filtered_content[i + 1].strip()
            merged_content.append(merged_line)

    # Step 4: Replace spaces with space and comma
    cleaned_content = []
    for line in merged_content:
        parts = line.split()
        if len(parts) >= 5 and parts[0] == 'Dataset' and parts[2] == 'Epoch':
            dataset_id = parts[1].strip(',')
            epoch_id = parts[4].strip(',')
            value = parts[5].strip(',')
            cleaned_line = f"{dataset_id},{epoch_id},{value}"
            cleaned_content.append(cleaned_line)

    # Step 5: Write the cleaned content to a temporary file for easier handling with pandas
    temp_file_path = '/mnt/data/temp_tabmlp_meta_learn_openml.a0802.1737262.tmp'
    with open(temp_file_path, 'w') as file:
        file.write("".join(cleaned_content))

    # Step 6: Load the cleaned content into a DataFrame
    cleaned_data = pd.read_csv(temp_file_path, header=None, names=['Dataset ID', 'Epoch ID', 'Value'])

    # Step 7: Convert columns to appropriate data types
    cleaned_data['Dataset ID'] = cleaned_data['Dataset ID'].astype(int)
    cleaned_data['Epoch ID'] = cleaned_data['Epoch ID'].astype(int)
    cleaned_data['Value'] = cleaned_data['Value'].astype(float)

    # Step 8: Sort the data by 'Dataset ID' and then by 'Epoch'
    sorted_data = cleaned_data.sort_values(by=['Dataset ID', 'Epoch ID']).reset_index(drop=True)

    # Step 9: Write the sorted data to the output file
    sorted_data.to_csv(output_file_path, index=False, header=False)

# Define the input and output file paths
input_file_path = './tabmlp_meta_learn_openml.a0802.1737262.err'
output_file_path = '/mnt/data/sorted_tabmlp_meta_learn_openml.a0802.1737262.csv'

# Process the log file
process_log_file(input_file_path, output_file_path)
