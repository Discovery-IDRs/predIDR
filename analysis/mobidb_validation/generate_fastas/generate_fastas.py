"""Generate FASTAs from raw MobiDB files."""

import os

from src.utils import read_fasta

def extract_features(fasta, label_name):
    features = {}
    for (header, seq) in fasta:
        accession = header.split('|')[0][1:]  # Trim >
        if accession not in features:
            features[accession] = {}
        if 'sequence' in header:
            features[accession]['seq_header'] = header
            features[accession]['seq'] = seq
        if label_name in header:
            features[accession]['label_header'] = header
            features[accession]['label'] = seq
    return features


curated_fasta = read_fasta('../../../data/MobiDB/2020_09/curated-disorder-merge.fasta')
missing_fasta = read_fasta('../../../data/MobiDB/2020_09/derived-missing_residues-th_90.fasta')
mobile_fasta = read_fasta('../../../data/MobiDB/2020_09/derived-mobile-th_90.fasta')

curated_features = extract_features(curated_fasta, 'curated-disorder-merge')
missing_features = extract_features(missing_fasta, 'derived-missing_residues-th_90')
mobile_features = extract_features(mobile_fasta, 'derived-mobile-th_90')

accessions = set(curated_features) | set(missing_features) | set(mobile_features)

if not os.path.exists('out/'):
    os.mkdir('out/')

with open('out/mobidb_seqs.fasta', 'w') as seqs_file, open('out/mobidb_labels.fasta', 'w') as labels_file:
    for accession in sorted(accessions):
        # Get seq and disorder labels
        if accession in curated_features:
            features = curated_features[accession]
            seq_header, label_header = features['seq_header'], features['label_header']
            seq, label = features['seq'], features['label']
        elif (accession in missing_features) and (accession not in mobile_features):
            features = missing_features[accession]
            seq_header, label_header = features['seq_header'], features['label_header']
            seq, label = features['seq'], features['label']
        elif (accession not in missing_features) and (accession in mobile_features):
            features = mobile_features[accession]
            seq_header, label_header = features['seq_header'], features['label_header']
            seq, label = features['seq'], features['label']
        else:  # Otherwise must be in both
            # Unpack features
            features1 = missing_features[accession]
            seq_header1, label_header1 = features1['seq_header'], features1['label_header']
            seq1, label1 = features1['seq'], features1['label']
            features2 = mobile_features[accession]
            seq_header2, label_header2 = features2['seq_header'], features2['label_header']
            seq2, label2 = features2['seq'], features2['label']
            if seq_header1 != seq_header2:
                raise RuntimeError('Headers do not match between missing and mobile annotations.')
            if seq1 != seq2:
                raise RuntimeError('Sequences do not match between missing and mobile annotations.')

            # Merge labels
            syms = []
            for sym1, sym2 in zip(label1, label2):
                if sym1 == sym2:
                    syms.append(sym1)
                else:  # If not the same, one of the labels is 1
                    syms.append('1')
            seq_header, label_header = seq_header1, label_header1.replace('derived-missing_residues-th_90', 'derived-merge')
            seq, label = seq1, ''.join(syms)

        # Write labels to file
        seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)])
        labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)])

        seqs_file.write(f'{seq_header}\n{seqstring}\n')
        labels_file.write(f'{label_header}\n{labelstring}\n')

"""
DEPENDENCIES
../../../data/mobidb/2020_09/*
"""