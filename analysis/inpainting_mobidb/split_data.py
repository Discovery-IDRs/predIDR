#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import os
import random
import datetime;

from Bio import SeqIO


# In[2]:


# Load data 
fasta_seq = SeqIO.parse("../inpainting_mobidb/out/label.fasta", 'fasta')
fasta_label = SeqIO.parse("../inpainting_mobidb/out/unmasked_seq_file.fasta", 'fasta')


# In[3]:


# Ratio for splitting into validation, test and train 
train_ratio = .80 
validation_ratio = .10 
test_ratio = .10 


# In[4]:


# Create dictionary with key-value pair, "protein_accession|start_ind:last_ind" : ["protein_labels", "amino_acid_seq", "description"] 
protein_dict = {}

# Load amino acid sequences into dictionary from allseq.fasta 
for protein in fasta_seq:
    key = protein.id.split("|")[0] + "|" + protein.description.split("|")[-1]
    protein_dict[key] = [str(protein.seq)]

# Edit dictionary to include amino acid sequence, labels, and descriptions from alldisorder.fasta 
for protein in fasta_label:
    key = protein.id.split("|")[0] + "|" + protein.description.split("|")[-1]
    protein_dict[key] = protein_dict.get(key) + [str(protein.seq)] + [protein.description]
    #accession = protein.id.split("|")[0]
    #protein_dict[accession] = protein_dict.get(accession) + [str(protein.seq)] + [protein.description]


# In[5]:


# Data Shuffling

# 2D list with a each list inside of the list for each protein [["protein_label", "amino_acid_sequence", "description"],..]
protein_lst = list(protein_dict.values())

# Use random seed for shuffling
random.seed(7)
random.shuffle(protein_lst)

# Extract by index 
train_length = math.ceil(0.8*len(protein_lst))
test_length = math.ceil(0.1*len(protein_lst))

train = protein_lst[:train_length]
test = protein_lst[train_length:train_length+test_length]
validation = protein_lst[train_length+test_length:]  # Validation gets remainder if split is not even


# In[6]:


# Create out directory to put fasta files in
data_path = "out/"
if not os.path.exists(data_path):
        os.mkdir(data_path)


# In[7]:


# Method for creating files 

def write_fasta(lst, label_or_seq, file_name):
    with open(data_path + file_name, "w+") as fasta_file:
        for record in lst:
            if label_or_seq == "label":
                label_str = "\n".join(record[0][i:i+80] for i in range(0, len(record[0]), 80)) 
                fasta_file.write(">" + record[2] + "\n" + label_str + "\n")
            elif label_or_seq == "seq":
                seq_str = "\n".join(record[1][i:i+80] for i in range(0, len(record[1]), 80))
                fasta_file.write(">" + record[2] + "\n" + seq_str + "\n")
                
    ct = datetime.datetime.now()
    print(file_name + " created ", ct)


# In[8]:


write_fasta(validation, "label", "validation_label.fasta")


# In[9]:


write_fasta(validation, "seq", "validation_seq.fasta")


# In[10]:


write_fasta(test, "label", "test_label.fasta")


# In[11]:


write_fasta(test, "seq", "test_seq.fasta")


# In[12]:


write_fasta(train, "label", "train_label.fasta")


# In[13]:


write_fasta(train, "seq", "train_seq.fasta")

