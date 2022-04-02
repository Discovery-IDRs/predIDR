#!/usr/bin/env python
# coding: utf-8

# In[93]:


import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage
from Bio import SeqIO


# In[94]:


fasta_seq = SeqIO.parse('../mobidb_validation/generate_fastas/out/allseq.fasta', 'fasta')
fasta_disorder = SeqIO.parse('../mobidb_validation/generate_fastas/out/alldisorder.fasta', 'fasta')


# In[95]:


def get_segments(aa_seq, label_seq, segment_type, accession, description):
    #print(label_seq)
    slices = ndimage.find_objects(ndimage.label(label_seq)[0])
    #print(slices)
    ds = []
    for s in slices:
        segment = aa_seq[s[0]]  # Unpack 1-element slice tuple
        d = {'accession': accession, 'description': description, 
             'segment_type': segment_type, 'len': len(segment), 
             'slice': s}
        
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


# In[96]:


protein_seq_dict = {}
for protein in fasta_seq:
    protein_seq_dict[protein.id.split("|")[0]] = str(protein.seq)


# In[97]:


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
#print(df1)


# In[98]:


#df1


# # Length Distribution of Disordered Regions in Proteins

# In[99]:


disorder = df1[df1['segment_type'] == 'D']
plt.hist(disorder['len'], bins=50)
plt.yscale('log')
plt.ylabel('Number of entries')
plt.xlabel('Number of Amino Acids')
plt.title('Length of Disordered Regions')


# ## Length of Disordered Regions Upper Limit

# Length of the disordered regions drops off significantly after about a length of 90, making the upper limit of the length of our data amino acid sequence be 180, because we want >50% unmasked (amino acid sequences of the ordered regions). 

# In[100]:


disless100 = disorder[disorder['len'] <= 100]
plt.hist(disless100['len'])


# The cut off for a disordered region is more than 30 amino acid residues. And we want the max disordered region length to be 90 amino acids, as stated above. This gives us 5,619 proteins for our dataset

# In[101]:


dismore30_less90 = disorder.loc[(disorder['len'] >= 30) & (disorder['len'] <= 90)]
plt.hist(dismore30_less90['len'])


# In[102]:


dismore30_less90['accession'].nunique()


# In[103]:


#dismore30_less90


# In[111]:


# Boolean List of whether there is enough context for amino acid sequence 
enough_context = []
# variable of residue length desired 
residue_len = 180

# List of all possible protein acccession and slice objects
acc_lst = list(dismore30_less90['accession'])
slice_lst = list(dismore30_less90['slice'])

for i in range(0, len(acc_lst)):
    acc = acc_lst[i]
    # 
    _slice = slice_lst[i][0]
    full_seq = protein_seq_dict[acc]
    dis_len = len(full_seq[_slice])
    
    # max content needed, (this means that excluding some proteins that have enough context) --> underestimate 
    context_len = (residue_len - dis_len)//2 + 1 
    
    # checking if there is enough context on both sides of the protein 
    cond = ((_slice.start - context_len >= 0) and (_slice.stop + context_len <= len(full_seq) - 1))
    
    enough_context.append(cond)


# In[114]:


print('minimum number of entries in dataset: ' + str(sum(enough_context)))

