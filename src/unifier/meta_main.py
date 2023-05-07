import json
import os, sys
from meta_analysor import fetch_meta, match_meta

if __name__ == "__main__":
    args = sys.argv
    path = args[2]
    meta_dataset = {}
    universal_set = []
    path_list = set(os.listdir(path))
    dataset_list = set(filter(lambda x: False if "meta_" in x else True, path_list))
    for _jsonfile in dataset_list:
        if "meta_"+_jsonfile not in path_list:
            with open(''.join((path,_jsonfile))) as json_dataset:
                meta_dataset["dataset_name"] = _jsonfile.split(".json")[0]
                dataset = json.load(json_dataset)
                meta_dataset["meta_columns"] = fetch_meta(dataset)
                with open(''.join((path,"meta_"+_jsonfile)),"w+") as metafile:
                    json.dump(meta_dataset, metafile, indent=4)

    meta_list = set(os.listdir(path))-dataset_list
    for _jsonfile in meta_list:
        with open(''.join((path,_jsonfile))) as json_metaset:
            match_meta(universal_set, json.load(json_metaset))
    print([list(map(lambda x: x["col"]["name"], u_set["columns"])) for u_set in universal_set])