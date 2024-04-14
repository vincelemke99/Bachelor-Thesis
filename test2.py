import pandas as pd


def remove_columns_from_csv(input_file, output_file, columns_to_remove):
    # Load the data from the CSV file
    data = pd.read_csv(input_file)

    # Remove the specified columns
    data.drop(columns_to_remove, axis=1, inplace=True)

    # Save the updated data to a new CSV file
    data.to_csv(output_file, index=False)


# Example usage:
input_file = 'trainings_data.csv' # Path to the input CSV file
output_file = 'trainings_data3.csv'  # Path to the output CSV file
columns_to_remove = ['name', 'email', 'produkt_code_pl']  # List of columns to remove

remove_columns_from_csv(input_file, output_file, columns_to_remove)
