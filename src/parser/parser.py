import argparse
import parse_datasets as parse
import os
import json
from globals import datasets

# Below is defined to parse command line arguments.

parser = argparse.ArgumentParser(description = "Dataset Parser for DL Engine\n" + 
											"For usage: datasets['dataset_name'] will return the corresponding dataset dictionary gathered from the file.\n" +
										    "From the dataset dictionary, by using column names of the datasets as indexes, " +
										    "one can gather corresponding data points.", formatter_class = argparse.RawTextHelpFormatter)
required_arguments = parser.add_argument_group("Required Arguments")
required_arguments.add_argument("-path", type = str, help = "A path to the folder that contains datasets", required = True)
args = parser.parse_args()

# This function dumps altered datasets file to the directory given!

def write_outputs(path_to_dir):
	global datasets
	if not os.path.exists(path_to_dir):
		os.makedirs(path_to_dir)
		print("Given path did not exist and created successfully!")
	os.chdir(path_to_dir)
	dataset_index = 0
	for dataset in datasets:
		with open(dataset + ".json", "w") as file:
			json.dump(datasets[dataset], file, indent = 4)

# Main function links datasets to the global dictionary defined for future use.

def main(path_to_dir):
	global datasets
	current_dir = os.getcwd()
	os.chdir(path_to_dir) # Change working directory to the path given.
	print(f"Changed working directory to: {os.getcwd()}")
	dataset = parse.parse_dataset()
	dataset_cnt, dataset_names = 0, []
	for idx in range(len(dataset)):
		dataset_name = parse.get_key(dataset, idx)
		datasets[dataset_name] = dataset[dataset_name]
		dataset_cnt = dataset_cnt + 1
		dataset_names.append(dataset_name)
	print(f"Read {dataset_cnt} datasets,\ncorresponding file names are: {dataset_names}")
	print(f"Changing working directory to project directory [{current_dir}] again!")
	os.chdir(current_dir)
	print(f"Writing resultant datasets to the JSON files to the directory: [{current_dir}/outputs]")
	write_outputs(current_dir + os.sep + 'outputs')


if __name__ == "__main__":
	main(args.path)