"""Partition DisProt records into seen and unseen sets."""

import json
import os

# Use JSON module to load file
with open('../../../data/DisProt/JSON/DisProt.json') as file:
    data = json.load(file)['data']

# Partition records into seen and unseen sets
unseen = []
seen = []
for record in data:
    if record['released'] == '2020_06' or record['released'] == '2019_09':
        unseen.append(record['disprot_id'])
    else:
        seen.append(record['disprot_id'])

# Make output directory
if not os.path.exists('out/'):
    os.mkdir('out/')

with open('out/unseen.txt', 'w') as file:
    for disprot_id in unseen:
        file.write(disprot_id + '\n')

with open('out/seen.txt', 'w') as file:
    for disprot_id in seen:
        file.write(disprot_id + '\n')

# Print results
print('Total number of records:', len(data))
print('Number of unseen records:', len(unseen))
print('Number of seen records:', len(seen))

"""
OUTPUT
Total number of records: 1454
Number of unseen records: 63
Number of seen records: 1391

DEPENDENCIES
../../../data/DisProt/JSON/DisProt.json
"""