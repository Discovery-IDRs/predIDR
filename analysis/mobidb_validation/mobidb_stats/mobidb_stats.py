"""Calculate and plot various statistics related to MobiDB sequences and annotations."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage


def load_fasta(path):
    fasta = []
    with open(path) as file:
        line = file.readline()
        while line:
            if line.startswith('>'):
                header = line
                line = file.readline()

            seqlines = []
            while line and not line.startswith('>'):
                seqlines.append(line.rstrip())
                line = file.readline()
            seq = ''.join(seqlines)
            fasta.append((header, seq))
    return fasta


def get_segments(aa_seq, label_seq, segment_type, accession):
    slices = ndimage.find_objects(ndimage.label(label_seq)[0])
    ds = []
    for s in slices:
        segment = aa_seq[s[0]]  # Unpack 1-element slice tuple
        d = {'accession': accession, 'segment_type': segment_type, 'len': len(segment)}
        aa_counts = count_amino_acids(segment)
        d.update(aa_counts)
        ds.append(d)
    return ds


def count_amino_acids(aa_seq):
    aa_codes = ['D', 'E', 'H', 'K', 'R', 'N', 'Q', 'S', 'T', 'A',
                'I', 'L', 'M', 'V', 'F', 'W', 'Y', 'C', 'G', 'P',
                'O', 'U', 'B', 'Z', 'X', 'J']
    d = {aa: 0 for aa in aa_codes}
    for aa in aa_seq:
        d[aa] += 1
    return d


seqs_dict = {}
for header, seq in load_fasta('../generate_fastas/out/mobidb_seqs.fasta'):
    accession = header.split('|')[0][1:]  # Trim >
    seqs_dict[accession] = seq
labels_dict = {}
for header, seq in load_fasta('../generate_fastas/out/mobidb_labels.fasta'):
    accession = header.split('|')[0][1:]  # Trim >
    labels_dict[accession] = seq

rows = []
for accession, label in labels_dict.items():
    seq = seqs_dict[accession]
    disorder_labels = [sym == '1' for sym in label]
    order_labels = [sym == '0' for sym in label]

    # Disordered regions have the code 'D' and ordered regions have the code 'O'
    # The entire protein is added with the code 'P'
    ds_disorder = get_segments(seq, disorder_labels, 'D', accession)
    ds_order = get_segments(seq, order_labels, 'O', accession)
    ds_protein = get_segments(seq, [True for _ in range(len(seq))], 'P', accession)

    # Add ds to rows
    rows.extend(ds_disorder)
    rows.extend(ds_order)
    rows.extend(ds_protein)
df1 = pd.DataFrame(rows)

if not os.path.exists('out/'):
    os.mkdir('out/')

# 1 Length distributions
# Length distribution of proteins
protein1 = df1[df1['segment_type'] == 'P']
plt.hist(protein1['len'], bins=50)
plt.ylabel('Number of proteins')
plt.xlabel('Number of residues')
plt.savefig('out/hist_numprot-numres1.png')
plt.yscale('log')
plt.savefig('out/hist_numprot-numres1_log.png')
plt.close()

# Length distribution of proteins after removing outliers
protein2 = protein1[protein1['len'] < 15000]
plt.hist(protein2['len'], bins=50)
plt.ylabel('Number of proteins')
plt.xlabel('Number of residues')
plt.savefig('out/hist_numprot-numres2.png')
plt.yscale('log')
plt.savefig('out/hist_numprot-numres2_log.png')
plt.close()

# Length distribution of disordered segments
disorder = df1[df1['segment_type'] == 'D']
plt.hist(disorder['len'], bins=50)
plt.ylabel('Number of disordered segments')
plt.xlabel('Number of residues')
plt.savefig('out/hist_numdis-numres.png')
plt.yscale('log')
plt.savefig('out/hist_numdis-numres_log.png')
plt.close()

# Length distribution of ordered segments
order = df1[df1['segment_type'] == 'O']
plt.hist(order['len'], bins=50)
plt.ylabel('Number of ordered segments')
plt.xlabel('Number of residues')
plt.savefig('out/hist_numord-numres.png')
plt.yscale('log')
plt.savefig('out/hist_numord-numres_log.png')
plt.close()

# Short segment stats
short = disorder[disorder['len'] < 10]
long = disorder[disorder['len'] >= 10]

print('Number of short segments:', len(short))
print('Fraction of short of total segments:', len(short) / len(disorder))
print()
print('Number of unique "short" proteins:', short['accession'].nunique())
print('Fraction of unique "short" proteins of total proteins:', short['accession'].nunique() / disorder['accession'].nunique())
print()
print('Number of unique "long" proteins:', long['accession'].nunique())
print('Fraction of unique "long" proteins of total proteins:', long['accession'].nunique() / disorder['accession'].nunique())
print()
print('Number of "long" residues:', long['len'].sum())
print('Fraction of "long" residues of total residues:', long['len'].sum() / disorder['len'].sum())

# 2 Fraction disordered distribution
disorder_lengths = disorder.groupby('accession')['len'].sum().rename('D_len')
df2 = protein1[['accession', 'len']].merge(disorder_lengths, on='accession', how='left').fillna(0)
df2['D_frac'] = df2['D_len'] / df2['len']

plt.hist(df2['D_frac'], bins=50)
plt.ylabel('Number of proteins')
plt.xlabel('Fraction disordered')
plt.savefig('out/hist_numprot_fracdis.png')
plt.yscale('log')
plt.savefig('out/hist_numprot_fracdis_log.png')
plt.close()

# 3 Fraction ordered distribution
order_lengths = order.groupby('accession')['len'].sum().rename('O_len')
df2 = df2.merge(order_lengths, on='accession', how='left').fillna(0)
df2['O_frac'] = df2['O_len'] / df2['len']

plt.hist(df2['O_frac'], bins=50)
plt.ylabel('Number of proteins')
plt.xlabel('Fraction ordered')
plt.savefig('out/hist_numprot_fracord.png')
plt.yscale('log')
plt.savefig('out/hist_numprot_fracord_log.png')
plt.close()

# 4 Scatter of fraction disordered with number of disordered segments
D_segnum = disorder.groupby('accession').size().rename('D_segnum')
df2 = df2.merge(D_segnum, on='accession', how='left').fillna(0)

plt.scatter(df2['D_segnum'], df2['D_frac'], s=6, alpha=0.25, edgecolors='none')
plt.ylabel('Fraction disordered')
plt.xlabel('Number of disordered segments')
plt.savefig('out/scatter_fracdis_numdis1.png')
plt.close()

# Without outliers
df2_1 = df2[df2['D_segnum'] < 50]
plt.scatter(df2_1['D_segnum'], df2_1['D_frac'], s=6, alpha=0.25, edgecolors='none')
plt.ylabel('Fraction disordered')
plt.xlabel('Number of disordered segments')
plt.savefig('out/scatter_fracdis-numdis2.png')
plt.close()

# 5 Scatter of average length of disordered segments with number of disordered segments
df3 = disorder[['accession', 'len']].groupby('accession').mean().merge(D_segnum, on='accession')
plt.scatter(df3['D_segnum'], df3['len'], s=6, alpha=0.25, edgecolors='none')
plt.ylabel('Average length of disordered segments')
plt.xlabel('Number of disordered segments')
plt.savefig('out/scatter_numdis-lendis1.png')
plt.close()

# Without outliers
df3_1 = df3[df3['D_segnum'] < 50]
plt.scatter(df3_1['D_segnum'], df3_1['len'], s=6, alpha=0.25, edgecolors='none')
plt.ylabel('Average length of disordered segments')
plt.xlabel('Number of disordered segments')
plt.savefig('out/scatter_numdis-lendis2.png')
plt.close()

# 6 Number disordered segments distribution
plt.hist(df2['D_segnum'], bins=50)
plt.ylabel('Number of proteins')
plt.xlabel('Number of disordered segments')
plt.savefig('out/hist_numprot-numdis1.png')
plt.yscale('log')
plt.savefig('out/hist_numprot-numdis1_log.png')
plt.close()

# Without outliers
plt.hist(df2.loc[df2['D_segnum'] < 60, 'D_segnum'], bins=50)
plt.ylabel('Number of proteins')
plt.xlabel('Number of disordered segments')
plt.savefig('out/hist_numprot-numdis2.png')
plt.yscale('log')
plt.savefig('out/hist_numprot-numdis2_log.png')
plt.close()

# 7 Total fraction ordered and disordered residues
plt.pie([order['len'].sum(), disorder['len'].sum()], labels=['order', 'disorder'], autopct='%1.0f%%')
plt.title('Residues by label')
plt.savefig('out/pie_labels.png')
plt.close()

# 8 Amino acid distributions and enrichment
aa_codes = ['D', 'E', 'H', 'K', 'R', 'N', 'Q', 'S', 'T', 'A',
            'I', 'L', 'M', 'V', 'F', 'W', 'Y', 'C', 'G', 'P',
            'O', 'U', 'B', 'Z', 'X', 'J']
aa_counts = df1.loc[df1['segment_type'] == 'P', aa_codes].sum()
aa_fracs = aa_counts / aa_counts.sum()
plt.bar(aa_codes, aa_fracs)
plt.ylabel('Fraction composition in proteins')
plt.xlabel('Amino acid')
plt.savefig('out/bar_aacomp_prot.png')
plt.close()

aa_counts_disorder = df1.loc[df1['segment_type'] == 'D', aa_codes].sum()
aa_fracs_disorder = aa_counts_disorder / aa_counts_disorder.sum()
plt.bar(aa_codes, aa_fracs_disorder)
plt.ylabel('Fraction composition in disordered segments')
plt.xlabel('Amino acid')
plt.savefig('out/bar_aacomp_dis.png')
plt.close()

aa_fracs_delta = aa_fracs_disorder - aa_fracs
plt.bar(aa_codes, aa_fracs_delta)
plt.ylabel('Difference in fraction composition')
plt.xlabel('Amino acid')
plt.savefig('out/bar_aacomp_delta.png')
plt.close()

ns_codes = df1.loc[df1['segment_type'] == 'P', ['accession', 'O', 'U', 'B', 'Z', 'X', 'J']]  # Non-standard codes
ns_codes['sum'] = ns_codes[['O', 'U', 'B', 'Z', 'X', 'J']].sum(axis=1)
ns_codes[ns_codes['sum'] > 0].to_csv('out/ns_codes.tsv', sep='\t', index=False)

"""
OUTPUT
Number of short segments: 49037
Fraction of short of total segments: 0.66375646335851

Number of unique "short" proteins: 29232
Fraction of unique "short" proteins of total proteins: 0.8032755351597922

Number of unique "long" proteins: 16826
Fraction of unique "long" proteins of total proteins: 0.4623670687807425

Number of "long" residues: 952023
Fraction of "long" residues of total residues: 0.8381354753398681

DEPENDENCIES
../generate_fastas/generate_fastas.py
    ../generate_fastas/out/*
"""