#code for writing FASTA file containing all sequences in DisProt JSON
#and for writing FASTA file for each individual sequence in DisProt JSON

import json
import os
import numpy as np

#use JSON module to laod file
with open("../../../data/{path_to_disprot_json}.json") as file:
    disprot = json.load(file)

#define functions to remove unwanted information from data
def remove_key(key, list):
    """Input key name as string and list name to remove key from dictionary list entries."""
    for i in np.arange(0, len(list)):
       list[i].pop(key, None)

def remove_key_all(keys, list):
    """Input array of keys to remove keys from dictionary list entries."""
    for key in keys:
        remove_key(key, list)

#define information to be removed from data
keys_to_remove_from_disprot = np.array(['uniref50', 'taxonomy', 'uniref90', 'creator', 'name', 'ncbi_taxon_id',
                           'organism', 'uniref100', 'date', 'regions_counter', 'released', 'disorder_content',
                            'features', 'length', 'disprot_consensus', 'features', 'regions',])

#remove defined information from data
remove_key_all(keys_to_remove_from_disprot, disprot["data"])

#following creates FASTA file with all protein sequences in JSON
fasta_file = open("all_sequence.fasta", "w")
for i in np.arange(len(disprot["data"])):
    fasta_file.write(">" + disprot["data"][i]["disprot_id"] + " " + disprot["data"][i]["acc"]
                     + "\n" + disprot["data"][i]["sequence"] + "\n" + "\n")
fasta_file.close()

#define function to create FASTA file of individual protein sequence
def fasta_file_writer(index):
    """Input index of protein sequence to create fasta file of specified sequence"""
    fasta_file = open(disprot["data"][index]["disprot_id"] + "sequence.fasta", "w")
    fasta_file.write(">" + disprot["data"][i]["disprot_id"] + " " + disprot["data"][i]["acc"]
                     + "\n" + disprot["data"][i]["sequence"])
    fasta_file.close()

#creating FASTA file for each individual protein sequence
for i in np.arange(len(disprot["data"])):
    fasta_file_writer(i)
