"""Exploration of MobiDB sequences and labels to establish length parameters for target and context."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage
from Bio import SeqIO


def get_segments(seq, labels, segment_type, accession, description):
    """
    Return list of dictionaries for each segment with parameters
           [ { 'accession': accession id, 'start': slice.start, 'stop': slice.stop', 'segment_type': segment_type,
               'segment_length': len(segment), 'seq_length': len(seq), 'description': description,
               'A': number of A in eq, 'R': number of R in seq ... }, ...]

    :param seq: string amino acid sequence
    :param labels: boolean list of whether disordered or ordered (0 = order, 1 = disorder)
    :param segment_type: 'D': disordered sequence 'O': ordered sequence 'P': entire protein
    :param accession: string accession id
    :param description: string description
    :return: list of dictionaries containing parameters for each segment_type sequence in entire protein
    """
    slices = ndimage.find_objects(ndimage.label(labels)[0])  # Finds contiguous intervals of True labels as slices

    rows = []
    for s, in slices:  # Unpack 1-element slice tuple
        segment = seq[s]
        row = {'accession': accession, 'start': s.start, 'stop': s.stop, 'segment_type': segment_type,
               'segment_length': len(segment), 'seq_length': len(seq), 'description': description}

        # Merge dictionary of amino acid counts to record dictionary
        counts = count_amino_acids(segment)
        row.update(counts)

        rows.append(row)

    return rows


def count_amino_acids(seq):
    """
    Return dictionary of counts of symbols in amino acid sequence in given seq

    :param seq: string of amino acid symbols
    :return: dictionary of the count of amino acids
    """
    counts = {sym: 0 for sym in alphabet}
    for sym in seq:
        counts[sym] += 1
    return counts


alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
            'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U', 'B', 'Z', 'X', 'J']

# Load data
fasta_seqs = SeqIO.parse('../../mobidb_validation/format_seqs/out/mobidb_seqs.fasta', 'fasta')
fasta_labels = SeqIO.parse('../../mobidb_validation/format_seqs/out/mobidb_labels.fasta', 'fasta')

# Create dictionary with key-value pair, "accession" : "amino acid sequence"
seq_records = {}
for record in fasta_seqs:
    accession = record.id.split("|")[0]
    seq_records[accession] = str(record.seq)

rows = []
for record in fasta_labels:
    # Get record information
    accession = record.id.split("|")[0]
    description = record.description.split("|")[-1]
    seq = seq_records[accession]
    labels = str(record.seq)

    # Lists of booleans if label is ordered or disordered
    labels_disorder = [sym == '1' for sym in labels]
    labels_order = [sym == '0' for sym in labels]
    labels_protein = [True for _ in range(len(labels))]

    # Disordered regions have the code 'D' and ordered regions have the code 'O'
    # The entire protein is added with the code 'P'
    rows_disorder = get_segments(seq, labels_disorder, 'D', accession, description)
    rows_order = get_segments(seq, labels_order, 'O', accession, description)
    rows_protein = get_segments(seq, labels_protein, 'P', accession, description)

    # Add to rows
    rows.extend(rows_disorder)
    rows.extend(rows_order)
    rows.extend(rows_protein)


if not os.path.exists('out/'):
    os.mkdir('out/')

# Create dataframe with information of disordered segment, ordered segments, and full protein
df1 = pd.DataFrame(rows)

# Length Distribution of Disordered Regions in Proteins
disorder = df1[df1['segment_type'] == 'D']
plt.hist(disorder['segment_length'], bins=50)
plt.yscale('log')
plt.ylabel('Number of regions')
plt.xlabel('Number of residues')
plt.title('Length distribution of disordered regions')
plt.savefig('out/length_disorder.png')
plt.close()

# Length of the disordered regions drops off significantly after about a length of 90, making the upper limit of the
# length of our data amino acid sequence be 180, because we want >50% unmasked (amino acid sequences of the ordered
# regions).
filtered1 = disorder[disorder['segment_length'] <= 100]
plt.hist(filtered1['segment_length'])
plt.ylabel('Number of regions')
plt.xlabel('Number of residues')
plt.title('Length distribution of disordered regions')
plt.savefig('out/length_upper_limit_disorder.png')
plt.close()

# The cut off for a disordered region is more than 30 amino acid residues. And we want the max disordered region
# length to be 90 amino acids, as stated above.
lower_limit = 30
upper_limit = 90
total_length = 180

filtered2 = disorder[(disorder['segment_length'] >= lower_limit) & (disorder['segment_length'] <= upper_limit)]
plt.hist(filtered2['segment_length'])
plt.ylabel('Number of regions')
plt.xlabel('Number of residues')
plt.title('Length distribution of disordered regions')
plt.savefig('out/length_lower_upper_limit_disorder.png')
plt.close()

# Number of unique proteins in dataset
# This gives us 5,619 proteins for our dataset
filtered2['accession'].nunique()

valid_segments = []  # List of accessions and slices where there is enough context
for row in filtered2.itertuples():
    length = row.stop - row.start

    context_length = (total_length - length) // 2
    remainder_length = (total_length - length) % 2
    start = row.start - context_length
    stop = row.stop + context_length + remainder_length

    # Check if there is enough context on both sides of the sequence
    if (start >= 0) and (stop <= row.seq_length):
        valid_segments.append((row.accession, start, stop))

print(f'Number of segments in data set: {len(valid_segments)}')
print(f'Number of unique accessions in data set: {len({accession for accession, _, _ in valid_segments})}')
