import csv
import os

# Below is for getting corresponding key to an index value provided.

def get_key(dictionary, index):
	for i, key in enumerate(dictionary.keys()):
		if i == index:
			return key

# Dataset parser

def parse_dataset():
	try:
		datasets = {}
		file_count = 0
		for file in os.listdir():
			if file.endswith(".csv"): # Check if current file is a csv file
				file_name = f"{file}"
				dataset_name = file_name.split(".")[0] # Get filename to use it as a key to our dataset.
				data = {}
				with open(file_name) as dataset: 
					file_being_read = csv.reader(dataset, delimiter = ",") # Read csv file
					is_column = False
					for row in file_being_read: 
						if not is_column: # First row is column names, which is containing the keys for our dataset.
							for index in range(1, len(row)):
								data[row[index]] = [] # Defining corresponding lists for dataset columns.
							is_column = True
						else:
							for index in range(len(data)):
								data[get_key(data, index)].append(row[index + 1]) # Acquiring key and appending data points to the corresponding key.
				datasets[dataset_name] = data
				file_count = file_count + 1
		if file_count == 0:
			raise ValueError("No datasets found!")
		return datasets
	except ValueError as error:
		exit(error)
