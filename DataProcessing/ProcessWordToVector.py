__author__ = 'Jonathan Simon'

import os
from lxml import etree

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

