import json
import os
import numpy as np

# Use JSON module to load file
with open('DisProt_raw.json') as file:
    disprot = json.load(file)

#define functions for removing information from data
def remove_key(key,lst):
    """Input key name as string to remove key from each dictionary entry"""
    for i in np.arange(0, len(lst)):
        lst[i].pop(key,None)

def remove_key_all(keys,lst):
	for key in keys:
		remove_key(key,lst)

#define information to be removed from data
keys_to_remove_from_disprot = np.array(['uniref50', 'taxonomy', 'uniref90', 'creator', 'name', 'ncbi_taxon_id',
                           'organism', 'uniref100', 'date', 'regions_counter', 'disorder_content',
                            'features', 'length', 'disprot_consensus', 'features', 'regions',])
remove_key_all(keys_to_remove_from_disprot,disprot['data'])

#Make an array of all released dates
released_dates = []
for dic in disprot['data']:
    released_dates.append(dic['released'])
released_dates

#Eliminate the repeated dates
released_dates = list(set(released_dates))
released_dates

#Make the dict grouped by dates --> Not needed
#disprot['data'][i].get('released') outputs the date
first =[] #2016
fourth =[] #2020
second =[] #2018
third =[] #2019
for i in np.arange(0,len(disprot['data'])):
    if disprot['data'][i].get("released") == released_dates[0]:
        first.append(disprot['data'][i])
    elif disprot['data'][i].get("released") == released_dates[2]:
        fourth.append(disprot['data'][i])
    elif disprot['data'][i].get("released") == released_dates[3]:
        second.append(disprot['data'][i])
    else:
        third.append(disprot['data'][i])

#Sort the data into seen and unseen
# seen is any data released before 2019_09
# unseen is any data released 2020_06 or 2019_09
seen =[]
unseen =[] 
for i in np.arange(0,len(disprot['data'])):
    if disprot['data'][i].get("released") == '2020_06' or disprot['data'][i].get("released")== '2019_09' :
        unseen.append(disprot['data'][i])
    else:
        seen.append(disprot['data'][i])

#remove 'released' from data for file conversion
remove_key('released',unseen)
remove_key('released',seen)

#creates fasta file containing all sequences in unseen
fasta_file_unseen= open("all_sequence_unseen.fasta", "w")
for i in np.arange(len(unseen)):
    fasta_file_unseen.write(">" + unseen[i]["disprot_id"] + " " + unseen[i]["acc"]
                     + "\n" + unseen[i]["sequence"] + "\n" + "\n")
fasta_file_unseen.close()

#creates fasta file containing all sequences in seen
fasta_file_seen = open("all_sequence_seen.fasta", "w")
for i in np.arange(len(seen)):
    fasta_file_seen.write(">" + seen[i]["disprot_id"] + " " + seen[i]["acc"]
                     + "\n" + seen[i]["sequence"] + "\n" + "\n")
fasta_file_seen.close()

#creates fasta file for each individual sequence in unseen
def fasta_file_writer(index):
    """Input index of protein sequence to create fasta file of specified sequence"""
    fasta_file_unseen = open(unseen[index]["disprot_id"] + "unseen_sequence.fasta", "w")
    fasta_file_unseen.write(">" + unseen[i]["disprot_id"] + " " + unseen[i]["acc"]
                     + "\n" + unseen[i]["sequence"])
    fasta_file_unseen.close()

#creating FASTA file for each individual protein sequence
for i in np.arange(len(unseen)):
    fasta_file_writer(i)

#creates fasta file for each individual sequence in seen
def fasta_file_writer(index):
    """Input index of protein sequence to create fasta file of specified sequence"""
    fasta_file_seen = open(seen[index]["disprot_id"] + "seen_sequence.fasta", "w")
    fasta_file_seen.write(">" + seen[i]["disprot_id"] + " " + seen[i]["acc"]
                     + "\n" + seen[i]["sequence"])
    fasta_file_seen.close()

#creating FASTA file for each individual protein sequence
for i in np.arange(len(seen)):
    fasta_file_writer(i)
