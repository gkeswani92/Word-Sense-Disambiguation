'''
Created on Oct 17, 2015

@author: gaurav
'''

from collections import OrderedDict
from lxml import etree
import pandas as pd
import json
import os
import csv

dir_path      = os.path.dirname(__file__) + '/'
training_file = 'DataFiles/training-data.data'
test_file     = 'DataFiles/test-data.data'
word2vec_file = 'DataFiles/word_vector_subset.pkl'
feature_vec   = 'DataFiles/feature_vector.json'
predictions   = 'DataFiles/predictions.csv'

def initializeXMLParser(path):
    '''
        Takes in the path of the XML file and returns the xml parser for the file
    '''
    parser        = etree.XMLParser(recover=True)
    training_tree = etree.parse(path, parser=parser)
    root          = training_tree.getroot()
    return root

def readWordToVector():
    '''
        Reads in the word to vector data frame that was created from the google 
        distribution for our project
    '''
    word_vector_subset = pd.read_pickle(dir_path + word2vec_file)
    return word_vector_subset

def readContextVectorData():
    f = open(dir_path + feature_vec)
    return json.load(f, object_pairs_hook=OrderedDict)

def saveContextVectorData(context_feature_data):
    f = open(dir_path + feature_vec, "w")
    json.dump(context_feature_data, f)
    f.close()

def pprint(myDict):
    print(json.dumps(myDict, indent = 4, sort_keys = True))
    
def savePredictionsToCSV(test_predictions):
    f = open(dir_path + predictions, "w")
    writer = csv.writer(f)
    writer.writerows([['Id','Prediction']])
    writer.writerows(test_predictions)
    f.close()
