"""Extract relevant fields from raw DisProt JSON file."""

import json
import os

# Use JSON module to load file
with open('../../../data/{path_to_disprot_json}.json') as file:
    disprot = json.load(file)  # Only loads "file-like" objects, so we need to open the file first

# Extract the relevant fields
# The follow is some temporary code with hints to help you get started
# Once you have a grasp of the internal structure of data, replace this section with your actual code
# Don't worry about deleting it! The magic of Git means we can always get it back

# The JSON module loads the DisProt data as a series of nested lists and dictionaries
# For example if we can the following
type(disprot)
# Python should tell us that disprot is a dictionary
# To see what's in this dictionary, we can execute
disprot.keys()
# You should see that this dictionaries contains two keys, 'data' and 'size'
# Let's see what size is
disprot['size']
# Python will tell us the value associated with 'size', is 1454; this is probably how many entries are in disprot
# Let's see what data is
disprot['data']
# You should get a bunch of output--the data key must contain everything else in the file
# Let's see what exactly this object is
type(disprot['data'])
# Ah, so it's a list. Let's see what exactly this list is made of. We can get the first object with slicing syntax
type(disprot['data'][0])
# We have a dictionary again! Let's see what's inside this dictionary
disprot['data'][0].keys()
# Finally it looks like we've reached the level of individual entries! Let's do a basic sanity check
len(disprot['data'])
# The length matches what was stored in the 'size' key at the top-level
# This confirms our suspicion that was the number of proteins in DisProt and that each key in disprot['data'] represents
# an individual entry
# I also know that Disprot has about ~1500 proteins, so I'm using a little bit of outside information to guide my
# reasoning
# So at this point, you should have a rough idea of how to manipulate these objects
# The name of the game now is to use these various fields to extract the sequences and the labels for each
# We'll want to include some metadata like labels so we can link our sequences back to the original entries in the raw
# data
# You should also consider what easily accessible output format you want to use to store your extracted fields
# It seems like we have sequences of amino acids or labels and metadata associated with those sequences
# Have you come across any formats like this?

# Make output directory
if not os.path.exists('out/'):
    os.mkdir('out/')

# Save extracted fields as file(s) in out

"""
DEPENDENCIES
../../../data/{path_to_disprot_json}.json
"""