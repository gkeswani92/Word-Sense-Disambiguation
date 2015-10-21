__author__ = 'Jonathan Simon'

import os
from lxml import etree
import pandas as pd

dir_path = os.path.dirname(__file__) + '/'
training_file = 'training-data.data'
test_file = 'test-data.data'

parser = etree.XMLParser(recover=True)

training_tree = etree.parse(dir_path+training_file, parser=parser)
training_root = training_tree.getroot()

test_tree = etree.parse(dir_path+test_file, parser=parser)
test_root = test_tree.getroot()

# # Contexts are broken up by the "<head>" tags indicating the target word
# # For example, to see the entirety of the first context:
# print training_root[0][0][1].text,      # <-- context preceding target word
# print training_root[0][0][1][0].text,   # <-- target word
# print training_root[0][0][1][0].tail    # <-- context following target word

# Grab all contexts from all training and test files, and concatenate them.
# Then cross-reference against the word2vec dataset
aggregate_context = []

for word_type in training_root:
    for word_instance in word_type:
        aggregate_context.extend(word_instance.find('context').text.split())
        aggregate_context.extend(word_instance.find('context').find('head').tail.split())

for word_type in test_root:
    for word_instance in word_type:
        aggregate_context.extend(word_instance.find('context').text.split())
        aggregate_context.extend(word_instance.find('context').find('head').tail.split())

aggregate_context_set = set(aggregate_context)

# Extract only the word vectors corresponding to words occuring in our contexts
# (~50k out of 3mil --> 1/60th the size)
path_to_word_vectors = '/Users/Macbook/Documents/Data_Analysis_and_ML_Projects/word2vec_Analogical_Chaining/Data/GoogleNews-vectors-negative300.txt'
w2v_dict = {}
is_header_line = True
with open(path_to_word_vectors) as infile:
    for line in infile:
        split_line = line.strip().split(' ')
        if is_header_line:
            is_header_line = False
        elif split_line[0] in aggregate_context_set:
            w2v_dict.update({split_line[0]: map(float,split_line[1:301])})

w2v_df = pd.DataFrame(data=w2v_dict)
w2v_df.to_csv(dir_path+'word_vector_subset.csv')
w2v_df.to_pickle(dir_path+'word_vector_subset.pkl') # csv file is too large, pickle it instead