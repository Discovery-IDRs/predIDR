#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from Bio import SeqIO


# In[ ]:


fasta_seq = SeqIO.parse('../mobidb_validation/generate_fastas/out/allseq.fasta', 'fasta')
fasta_disorder = SeqIO.parse('../mobidb_validation/generate_fastas/out/alldisorder.fasta', 'fasta')


# In[ ]:


def get_segments(aa_seq, label_seq, segment_type, accession, description):
    #print(label_seq)
    slices = ndimage.find_objects(ndimage.label(label_seq)[0])
    #print(slices)
    ds = []
    for s in slices:
        segment = aa_seq[s[0]]  # Unpack 1-element slice tuple
        d = {'accession': accession, 'description': description, 
             'segment_type': segment_type, 'len': len(segment), 
             'index': (s[0].start, s[0].stop)}
        
        aa_counts = count_amino_acids(segment)
        d.update(aa_counts)
        ds.append(d)
    return ds


def count_amino_acids(aa_seq):
    aa_codes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
                'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 
                'Y', 'V', 'O', 'U', 'B', 'Z', 'X', 'J']
    d = {aa: 0 for aa in aa_codes}
    for aa in aa_seq:
        d[aa] += 1
    return d


# In[ ]:


protein_seq_dict = {}
for protein in fasta_seq:
    protein_seq_dict[protein.id.split("|")[0]] = str(protein.seq)


# In[ ]:


rows = []
for protein in fasta_disorder:
    dis_labels = [s == '1' for s in protein.seq]
    ord_labels = [s == '0' for s in protein.seq]

    accession = protein.id.split("|")[0]
    description = protein.description.split("|")[-1]
    aa_seq = protein_seq_dict[accession]
    
    # Disordered regions have the code 'D' and ordered regions have the code 'O'
    # The entire protein is added with the code 'P'
    ds_dis = get_segments(aa_seq, dis_labels, 'D', accession, description)
    ds_ord = get_segments(aa_seq, ord_labels, 'O', accession, description)
    ds_all = get_segments(aa_seq, [True for _ in range(len(aa_seq))], 'P', accession, description)

    # Add ds to rows
    rows.extend(ds_dis)
    rows.extend(ds_ord)
    rows.extend(ds_all)
df1 = pd.DataFrame(rows)
print(df1)


# In[ ]:


df1


# # Length Distribution of Disordered Regions in Proteins

# In[ ]:


disorder = df1[df1['segment_type'] == 'D']
plt.hist(disorder['len'], bins=50)
plt.yscale('log')
plt.ylabel('Number of entries')
plt.xlabel('Number of Amino Acids')
plt.title('Length of Disordered Regions')


# ## Length of Disordered Regions Upper Limit

# Length of the disordered regions drops off significantly after about a length of 90, making the upper limit of the length of our data amino acid sequence be 180, because we want >50% unmasked (amino acid sequences of the ordered regions). 

# In[ ]:


disless100 = disorder[disorder['len'] <= 100]
plt.hist(disless100['len'])


# The cut off for a disordered region is more than 30 amino acid residues. And we want the max disordered region length to be 90 amino acids, as stated above. This gives us 5,619 proteins for our dataset

# In[ ]:


dismore30 = disless100[disless100['len'] >= 30]
plt.hist(dismore30['len'])
disless90 = dismore30[dismore30['len'] <= 90] 
plt.hist(disless90['len'])


# In[ ]:


disless90

