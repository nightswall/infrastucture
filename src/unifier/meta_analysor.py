import datetime
import copy
import sys
import re
import math
from difflib import SequenceMatcher

# dateformat = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d.%m.%Y", "%Y.%m.%d", "%d %m %Y", "%Y %m %d"]
# datetimehmformat = [_tf+" %H:%M" for _tf in dateformat]
# datetimehmsformat = [_tf+":%S" for _tf in datetimehmformat]
# datetimehmsmformat = [_tf+".%f" for _tf in datetimehmsformat]

NAME_WEIGHT = 30
NUMBER_WEIGHT = 23
DATE_WEIGHT = 23
STRING_WEIGHT = 23
MIN_THRESH = 0.25

dateformatre = [
    "^[0-9]{1,2}\\/[0-9]{1,2}\\/[0-9]{4}", "^[0-9]{4}\\/[0-9]{1,2}\\/[0-9]{1,2}",
    "^[0-9]{1,2}\\.[0-9]{1,2}\\.[0-9]{4}", "^[0-9]{4}\\.[0-9]{1,2}\\.[0-9]{1,2}",
    "^[0-9]{1,2}\\s[0-9]{1,2}\\s[0-9]{4}", "^[0-9]{4}\\s[0-9]{1,2}\\s[0-9]{1,2}",
    "^[0-9]{1,2}\\-[0-9]{1,2}\\-[0-9]{4}", "^[0-9]{4}\\-[0-9]{1,2}\\-[0-9]{1,2}"
]

repat = ["/", ".", " ", "-"]

common_types = {
    "STR": {
        "occurrence": 0,
        "keywords": {}
    },
    "NUMBER": {
        "occurrence": 0,
        "min": sys.maxsize,
        "max": -sys.maxsize,
        "avg": None
    },
    "DATE": {
        "occurrence": 0,
        "min": sys.maxsize,
        "max": -sys.maxsize,
        "min_str": None,
        "max_str": None,
    }
}

def datetime_reformatter(datetime: str) -> str:
    for pattern in dateformatre:
        res = re.search(pattern, datetime)
        if res:
            _date=res.group(0)
            _time=re.sub(pattern, "", datetime)
            index = dateformatre.index(pattern)
            _date=_date.split(repat[index//2])
            if index%2 == 0:
                _date=_date[::-1]
            _date="-".join(_date)
            return _date+_time

def datatype_decider(data: str) -> any:
    data_lower: str = data.lower()
    datatype_pred: str = ""
    date_candidate = None
    converted_data = data
    try:
        int(data_lower)
        datatype_pred+=("I")
    except:
        pass

    try:
        float(data_lower)
        datatype_pred+=("F")
        if 1_000_000_000 < float(data_lower) < datetime.datetime.utcnow().timestamp():
            datatype_pred+=("u")
    except:
        pass
    
    if not datatype_pred:
        try:
            date_candidate = datetime_reformatter(data)
            datetime.datetime.fromisoformat(date_candidate)
            datatype_pred+=("DT")
        except:
            pass

    if not datatype_pred:
        datatype_pred = "STR"

    else:
        if datatype_pred == "IFu":
            converted_data =  int(data)
            datatype_pred = "DATE"

        elif datatype_pred == "IF":
            converted_data = float(data)
            datatype_pred = "NUMBER"

        elif datatype_pred == "Fu":
            converted_data = float(data)
            datatype_pred = "DATE"

        elif datatype_pred == "F":
            converted_data = float(data)
            datatype_pred = "NUMBER"

        elif datatype_pred == "DT":
            converted_data = datetime.datetime.fromisoformat(date_candidate).timestamp()
            datatype_pred = "DATE"

        elif datatype_pred == "DTDT":
            print(data_lower)

        else:
            print("Awkward: ", data)

    return converted_data, datatype_pred

def fetch_meta(dataset: dict) -> dict:
    dataset_meta = {}
    for column in dataset.keys():
        column_meta = copy.deepcopy(common_types)
        for data in dataset[column]:
            cdata, pred = datatype_decider(data)
            column_meta[pred]["occurrence"]+=1
            if pred in ("INT", "NUMBER", "DATE"):
                column_meta[pred]["min"] = min(column_meta[pred]["min"], cdata)
                column_meta[pred]["max"] = max(column_meta[pred]["max"], cdata)
            
            elif pred in ("STR"):
                if column_meta[pred]["keywords"].get(cdata, None):
                    column_meta[pred]["keywords"][cdata]+=1
                else:
                    column_meta[pred]["keywords"][cdata]=1

        for key in column_meta.keys():
            if not column_meta[key]["occurrence"]:
                column_meta[key] = None
        
        if column_meta['STR']:
            for key in column_meta['STR']['keywords']:
                column_meta['STR']['keywords'][key]=float("{:.3f}".format(column_meta['STR']['keywords'][key]/column_meta['STR']['occurrence']))
        
        if column_meta['DATE']:
            column_meta['DATE']['min_str'] = (datetime.datetime.fromtimestamp(column_meta['DATE']['min'])).isoformat()
            column_meta['DATE']['max_str'] = (datetime.datetime.fromtimestamp(column_meta['DATE']['max'])).isoformat()

        column_meta["name"] = column
        column_meta["number_of_rows"] = len(dataset[column])
        dataset_meta[column] = column_meta

    return dataset_meta

def name_similarity(column_name: str, column_set: list) -> float:
    ratio = 0
    for column_in_set in column_set:
        ratio = max(ratio, SequenceMatcher(None, column_name, column_in_set["col"]["name"]).ratio())
    return ratio

def number_similarity(column_number: dict, column_set: list) -> float:
    ratio = 0
    for column_in_set in column_set:
        # jaccard similarity (A n B)/(A u B)
        union = (min(column_number["min"], column_in_set["col"]["NUMBER"]["min"]) , max(column_number["max"], column_in_set["col"]["NUMBER"]["max"]))
        intersect = (max(column_number["min"], column_in_set["col"]["NUMBER"]["min"]), min(column_number["max"], column_in_set["col"]["NUMBER"]["max"]))
        if intersect[0] > intersect[1] or intersect[1]-intersect[0] == 0: # min>max, so no intersect
            ratio = 0
            continue
        ratio = max(ratio, (intersect[1]-intersect[0])/(union[1]-union[0]))
    return ratio

def str_similarity(column_str: dict, column_set: list) -> float:
    ratio = 0
    str_set = [k for j in ([i]*int(1000*column_str["keywords"][i]) for i in column_str["keywords"].keys()) for k in j]
    for column_in_set in column_set:
        tmp_str_set = [k for j in ([i]*int(1000*column_in_set["col"]["STR"]["keywords"][i]) for i in column_in_set["col"]["STR"]["keywords"].keys()) for k in j]
        union = str_set+tmp_str_set
        min_match = {i: min(column_str["keywords"][i],column_in_set["col"]["STR"]["keywords"][i]) for i in column_str["keywords"] if i in column_in_set["col"]["STR"]["keywords"]}
        intersect = [k for j in ([i]*int(1000*min_match[i]) for i in min_match.keys()) for k in j]
        ratio = max(ratio, len(intersect)/len(union))
    
    return ratio

def match_meta(universal_set: list, dataset: dict) -> list:
    if not universal_set:
        for column in dataset["meta_columns"]:
            match_set = [[1]]
            u_set = {"match_set": match_set, "columns": [{ "src": dataset["dataset_name"], "col": dataset["meta_columns"][column]}]}
            universal_set.append(u_set)
    else:
        for column in dataset["meta_columns"]:
            u_len = len(universal_set)
            column_similarity_ratio = {
                "name_sr": [0]*u_len,
                "number_sr": [0]*u_len,
                "date_sr": [0]*u_len,
                "str_sr": [0]*u_len,
                "overall_sr": [0]*u_len
            }

            overall_sr_divisor = NAME_WEIGHT

            if len(list(filter(lambda x: x!=None, dataset["meta_columns"][column].values()))) != 3:
                print("Corrupt column: ", column, dataset["meta_columns"][column])
                continue           

            for u_set in universal_set:
                u_index = universal_set.index(u_set)
                if dataset["dataset_name"] in list(map(lambda x: x["src"], u_set["columns"])):
                    column_similarity_ratio["overall_sr"][u_index] = -1 # Do not match columns that belogns same dataset
                    continue
                
                if dataset["meta_columns"][column]["DATE"] and u_set["columns"][0]["col"]["DATE"]:
                    column_similarity_ratio["date_sr"][u_index] = 1 # Try to unify all date raleted columns
                    overall_sr_divisor+=DATE_WEIGHT

                if dataset["meta_columns"][column]["NUMBER"] and u_set["columns"][0]["col"]["NUMBER"]:
                    column_similarity_ratio["number_sr"][u_index] = number_similarity(dataset["meta_columns"][column]["NUMBER"], u_set["columns"])
                    overall_sr_divisor+=NUMBER_WEIGHT
                
                if dataset["meta_columns"][column]["STR"] and u_set["columns"][0]["col"]["STR"]:
                    column_similarity_ratio["str_sr"].append( str_similarity(dataset["meta_columns"][column]["STR"], u_set["columns"]) )
                    overall_sr_divisor+=STRING_WEIGHT
                
                column_similarity_ratio["name_sr"][u_index] = name_similarity(dataset["meta_columns"][column]["name"], u_set["columns"])
                
                column_similarity_ratio["overall_sr"][u_index] =                    \
                    (column_similarity_ratio["name_sr"][u_index]*NAME_WEIGHT +      \
                    column_similarity_ratio["date_sr"][u_index]*DATE_WEIGHT +       \
                    column_similarity_ratio["number_sr"][u_index]*NUMBER_WEIGHT +   \
                    column_similarity_ratio["str_sr"][u_index]*STRING_WEIGHT)       \
                    / overall_sr_divisor
                             
                    
            max_match_ratio = max(column_similarity_ratio["overall_sr"])
            target_dytpe = list(filter(lambda x: dataset["meta_columns"][column][x] != None, dataset["meta_columns"][column]))[0]

            if max_match_ratio > MIN_THRESH:
                match_index = column_similarity_ratio["overall_sr"].index(max_match_ratio)
                if universal_set[match_index]["columns"][0]["col"][target_dytpe]:
                    universal_set[match_index]["columns"].append({ "src": dataset["dataset_name"], "col": dataset["meta_columns"][column]})
                    continue

            match_set = [[1]]
            universal_set.append({"match_set": match_set, "columns": [{ "src": dataset["dataset_name"], "col": dataset["meta_columns"][column]}]})
                
    return universal_set