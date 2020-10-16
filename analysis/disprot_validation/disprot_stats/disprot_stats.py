"""Plot various statistics associated with the entries in the DisProt database."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd

# Use JSON module to load file
with open('../../../data/DisProt/JSON/DisProt.json') as file:
    data = json.load(file)['data']

columns = ['length', 'disprot_id', 'released', 'disorder_content']

# Extract relevant fields from DisProt
rows = []
for record in data:
    row = {}
    for field in ['disprot_id', 'name', 'organism', 'length', 'released']:
        row[field] = record[field]
    row['taxonomy'] = record['taxonomy'][0]
    num_regions = 0
    num_residues = 0
    for region in record['disprot_consensus']['structural_state']:
        if region['type'] == 'D':
            num_regions += 1
            num_residues += region['start'] - region['end'] + 1  # Add 1 since endpoints included
    rows.append(row)
df = pd.DataFrame(rows)

# Make output directory
if not os.path.exists('out/'):
    os.mkdir('out/')

# Distribution of entries across taxonomic groups
counts = df['taxonomy'].value_counts()
plt.bar(counts.index, counts.values)
plt.xlabel('Domain')
plt.ylabel('Number of DisProt entries')
plt.savefig('out/bar_taxa.png')

"""
DEPENDENCIES
../../../data/DisProt/JSON/DisProt.json
"""