import numpy as np
import json
import os

#use JSON module to laod file
#note that I called the dictionary data
with open("../../../data/{path_to_disprot_json}.json") as file:
    data = json.load(file)

#function to remove key from each dictionary entry in json dictionary
#note that the dictionary is data
#be sure to change the dictionary name to whatever you are using if not using data
def remove_key(key):
    """Input key name as string to remove key from each dictionary entry"""
    for i in np.arange(0, data["size"]):
        data["data"][i].pop(key, None)

#specifies keys that we want to remove
keys_to_remove = np.array(['uniref50', 'taxonomy', 'uniref90', 'creator', 'name', 'ncbi_taxon_id','organism', 'uniref100', 'date', 'regions_counter'])

#removes specified keys
for key in keys_to_remove:
    remove_key(key)

#creates JSON of cleaned data
with open("../../../data/{path_to_cleaned_data_json}.json", "w") as file:
    json.dump(data, file)
