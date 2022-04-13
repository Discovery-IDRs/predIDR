"""Data exploration and data formatting"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage
import os
from Bio import SeqIO

def get_segments(aa_seq, label_seq, segment_type, accession, description):
    """
    Outputs list of dictionaries for each segment with parameters
           [ { 'accession': accession id, 'description': description, 'segment_type': segment_type,
           'len': length, 'slice': slice, 'A': number of A in aa_seq, 'R': number of R in aa_seq ... }, ...]

    :param aa_seq: string amino acid sequence
    :param label_seq: boolean list of whether disordered or ordered (0 = order, 1 = disorder)
    :param segment_type: 'D': disordered sequence 'O': ordered sequence 'P': entire protein
    :param accession: string accession id
    :param description: string description
    :return: list of dictionaries containing parameters for each segment_type sequence in entire protein
    """
    # Outputs the range of ordered/disordered sequences as a slice
    slices = ndimage.find_objects(ndimage.label(label_seq)[0])

    segments = []

    for s in slices:

        segment = aa_seq[s[0]]  # Unpack 1-element slice tuple
        record = {'accession': accession, 'description': description,
             'segment_type': segment_type, 'len': len(segment), 
             'slice': s}

        # Merging dictionary of amino acid counts to record dictionary
        aa_counts = count_amino_acids(segment)
        record.update(aa_counts)

        segments.append(record)

    return segments


def count_amino_acids(aa_seq):
    """
    Creates dictionary of counts of amino acid sequence in given aa_seq

    :param aa_seq: string of amino acid sequence
    :return: dictionary of the count of amino acids
    """
    aa_codes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
                'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 
                'Y', 'V', 'O', 'U', 'B', 'Z', 'X', 'J']
    d = {aa: 0 for aa in aa_codes}
    for aa in aa_seq:
        d[aa] += 1
    return d


# Load data
fasta_seq = SeqIO.parse("../../mobidb_validation/format_seqs/out/mobidb_seqs.fasta", "fasta")
fasta_disorder = SeqIO.parse("../../mobidb_validation/format_seqs/out/mobidb_labels.fasta", "fasta")


# Create dictionary with key-value pair, "accession" : "amino acid sequence"
protein_seq_dict = {}
for protein in fasta_seq:
    protein_seq_dict[protein.id.split("|")[0]] = str(protein.seq)

rows = []
for protein in fasta_disorder:
    # Lists of booleans if label is ordered or disordered
    dis_labels = [s == '1' for s in protein.seq]
    ord_labels = [s == '0' for s in protein.seq]

    # Obtaining amino acid sequence from protein_seq_dict
    accession = protein.id.split("|")[0]
    aa_seq = protein_seq_dict[accession]

    # Obtaining description for get_segments parameter
    description = protein.description.split("|")[-1]

    # Disordered regions have the code 'D' and ordered regions have the code 'O'
    # The entire protein is added with the code 'P'
    ds_dis = get_segments(aa_seq, dis_labels, 'D', accession, description)
    ds_ord = get_segments(aa_seq, ord_labels, 'O', accession, description)
    ds_all = get_segments(aa_seq, [True for _ in range(len(aa_seq))], 'P', accession, description)

    # Add ds to rows
    rows.extend(ds_dis)
    rows.extend(ds_ord)
    rows.extend(ds_all)

# Create dataframe with information of disordered segment, ordered segments, and full protein
df1 = pd.DataFrame(rows)

if not os.path.exists("out/"):
    os.mkdir("out/")

# Length Distribution of Disordered Regions in Proteins
disorder = df1[df1['segment_type'] == 'D']
plt.hist(disorder['len'], bins=50)
plt.yscale('log')
plt.ylabel('Number of entries')
plt.xlabel('Number of Amino Acids')
plt.title('Length of Disordered Regions')
plt.savefig('out/len_disorder.png')

# Length of the disordered regions drops off significantly after about a length of 90, making the upper limit of the
# length of our data amino acid sequence be 180, because we want >50% unmasked (amino acid sequences of the ordered
# regions).
disless100 = disorder[disorder['len'] <= 100]
plt.hist(disless100['len'])
plt.savefig('out/len_upper_limit_disorder.png')

# The cut off for a disordered region is more than 30 amino acid residues. And we want the max disordered region
# length to be 90 amino acids, as stated above.
dismore30_less90 = disorder.loc[(disorder['len'] >= 30) & (disorder['len'] <= 90)]
plt.hist(dismore30_less90['len'])
plt.savefig('out/len_lower_upper_limit_disorder.png')

# Number of unique proteins in dataset
# This gives us 5,619 proteins for our dataset
dismore30_less90['accession'].nunique()

# Boolean List of whether there is enough context for amino acid sequence 
enough_context = []
# variable of residue length desired 
residue_len = 180

# List of all possible protein accession and slice objects
acc_lst = list(dismore30_less90['accession'])
slice_lst = list(dismore30_less90['slice'])

for i in range(0, len(acc_lst)):
    acc = acc_lst[i]

    _slice = slice_lst[i][0]
    full_seq = protein_seq_dict[acc]
    dis_len = len(full_seq[_slice])
    
    # max content needed, (this means that excluding some proteins that have enough context) --> underestimate 
    context_len = (residue_len - dis_len)//2 + 1 
    
    # checking if there is enough context on both sides of the protein 
    cond = ((_slice.start - context_len >= 0) and (_slice.stop + context_len <= len(full_seq) - 1))
    
    enough_context.append(cond)


print('minimum number of entries in dataset: ' + str(sum(enough_context)))
