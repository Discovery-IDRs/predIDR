#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import scipy.ndimage as ndimage
from Bio import SeqIO


# In[2]:


# Set path of previously created fasta files of amino acid sequence and labels 
fasta_seq = SeqIO.parse('../mobidb_validation/generate_fastas/out/allseq.fasta', 'fasta')
fasta_disorder = SeqIO.parse('../mobidb_validation/generate_fastas/out/alldisorder.fasta', 'fasta')


# In[3]:


# Create dictionary with key-value pair, "protein_accession" : ["amino_acid_sequence", "protein_labels", "description"] 
protein_dict = {}
# Load amino acid sequences into dictionary from allseq.fasta 
for protein in fasta_seq:
    protein_dict[protein.id.split("|")[0]] = [str(protein.seq)]
# Edit dictionary to include amino acid sequence, labels, and descriptions from alldisorder.fasta 
for protein in fasta_disorder:
    accession = protein.id.split("|")[0]
    protein_dict[accession] = protein_dict.get(accession) + [str(protein.seq)] + [protein.description]


# In[4]:


# Create out directory to put fasta files in 
data_path = "out/"
if not os.path.exists(data_path):
        os.mkdir(data_path)


# In[5]:


# Create fasta file with labels and unmasked amino acid sequences 
labels_file = open(data_path + "label.fasta", "w+")
unmasked_seq_file = open(data_path + "unmasked_seq_file.fasta", "w+")


# In[6]:


# Create variables determined from inpainting_exploration.ipynb of upper and lower limit of length of disordered region
dis_lower_limit = 30
dis_upper_limit = 90
len_residue = 180

# List to store proteins in dataset that fulfil requirements to check the number of proteins
protein_lst = set()
not_protein_lst = set()

# Iterate through all proteins 
for protein_id in protein_dict:
    
    # Find the disordered regions of the protein
    label = protein_dict.get(protein_id)[1]
    dis_labels = [s == '1' for s in label]
    
    slices = ndimage.find_objects(ndimage.label(dis_labels)[0])
    
    for s in slices:
        len_seg = len(label[s[0]])
        # Checking to see if disordered region is of desired length as set from variables declared above 
        if len_seg >= dis_lower_limit and len_seg <= dis_upper_limit:
            # Calculating the index for the context of the disordered region
            len_context = (len_residue - len_seg) // 2
            
            len_remainder = (len_residue - len_seg) % 2 
            
            # only need index of disordered sequence
            start_ind = s[0].start - len_context 
            end_ind = s[0].stop + len_context
            
            output_labels = len_context*'0' + len_seg*'1' + '0'*(len_context + len_remainder)
            output_aaseq = protein_dict.get(protein_id)[0][slice(start_ind, end_ind + len_remainder)]

            
            # Writing the description and the labels/amino acid sequences of proteins that fits the desired length
            if len(output_labels) == len_residue and len(output_aaseq) == len_residue:

                protein_lst.add(protein_id)
                
                labels_file.write(">" + protein_dict.get(protein_id)[2] + "|" + str(start_ind) + ":" + str(end_ind) + "\n"
                                 + "\n".join([output_labels[i:i+80] for i in range(0, len(output_labels), 80)]) + "\n")
                
                unmasked_seq_file.write(">" + protein_dict.get(protein_id)[2] + "|" + str(start_ind) + ":" + str(end_ind) + "\n"
                                       + "\n".join([output_aaseq[i:i+80] for i in range(0, len(output_aaseq), 80)]) + "\n")
                
            elif len(output_aaseq) != len_residue:
                not_protein_lst.add(protein_id)


labels_file.close()
unmasked_seq_file.close()
print('len of dataset: ' + str(len(protein_lst)))
