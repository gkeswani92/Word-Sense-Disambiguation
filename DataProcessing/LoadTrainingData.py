'''
Created on Oct 13, 2015

@author: gaurav
'''

from collections import defaultdict
from numpy.linalg import norm
from lxml import etree
import pandas as pd
import os

dir_path      = os.path.dirname(__file__) + '/'
training_file = 'training-data.data'
test_file     = 'test-data.data'
word2vec_file = 'word_vector_subset.csv'

def initializeXMLParser(path):
    '''
        Takes in the path of the XML file and returns the xml parser for the file
    '''
    parser        = etree.XMLParser(recover=True)
    training_tree = etree.parse(path, parser=parser)
    root          = training_tree.getroot()
    return root

def getTrainingContextData():
    
    training_data = defaultdict(lambda: defaultdict(dict))
    
    #Initialising the xml parser for the training and test set
    training_root = initializeXMLParser(dir_path+training_file) 
    
    #Grabbing one word type at a time
    for word_type_xml in training_root:
        word_type = word_type_xml.attrib['item']
        
        #Grabbing the instance id and its list of senses
        for word_instance in word_type_xml:
            instance = word_instance.attrib['id']
            senses   = [answer.attrib['senseid'] for answer in word_instance.findall('answer')]
            pre_context  = word_instance.find('context').text.split()
            post_context = word_instance.find('context').find('head').tail.split()
            
            training_data[word_type]['training'][instance] = {"Sense":senses, "Pre-Context":pre_context, "Post-Context":post_context }
            break
        break
    return training_data

def getTestContextData(test_data):
    
    #Initialising the xml parser for the training and test set
    training_root = initializeXMLParser(dir_path + test_file) 
    
    #Grabbing one word type at a time
    for word_type_xml in training_root:
        word_type = word_type_xml.attrib['item']
        
        #Grabbing the instance id and its list of senses
        for word_instance in word_type_xml:
            instance = word_instance.attrib['id']
            pre_context  = word_instance.find('context').text.split()
            post_context = word_instance.find('context').find('head').tail.split()
            
            test_data[word_type]['test'][instance] = {"Pre-Context":pre_context, "Post-Context":post_context }
            break
        break
    return test_data

def readWordToVector():
    '''
        Reads in the word to vector data frame that was created from the google 
        distribution for our project
    '''
    word_vector_subset = pd.read_csv(dir_path + word2vec_file)
    return word_vector_subset

def makeFeatureVectorForWordInstance(context_data, word_vector_subset, window_size = 5):
    '''
        Creates the feature vector for each word instance by reading the word vectors
        from the word to vec data frame that we created
    '''
    for word_type, word_type_data in context_data.iteritems():
        for data_type, instance_details in word_type_data.iteritems():
            for instance, context_details in instance_details.iteritems():
                
                context        = getContextWordsinWindow(context_details, window_size)
                feature_vector = createFeatureVectorFromContext(context, word_vector_subset)
                context_data[word_type][data_type][instance].update({"Feature_Vector":feature_vector})
                
    print(context_data)            
    
def createFeatureVectorFromContext(context, word_vector_subset):
    '''
        Creates the feature vector from the google word2vec vectors depending on 
        the context words passed in
    '''
    token_vectors = word_vector_subset.ix[:,context]
    vectors_sum = token_vectors.sum(axis = 1)
    normalised_vectors = vectors_sum / norm(vectors_sum)
    normalised_vector_list = normalised_vectors.tolist()
    return normalised_vector_list
                  
def getContextWordsinWindow(context_details, window_size):
    '''
        Gets the appropriate context words from the pre context and the post 
        context depending on the window size that is passed in
    '''
    pre_context  = context_details['Pre-Context']
    post_context = context_details['Post-Context']    
    context      = []
    
    if len(pre_context) >= window_size and len(post_context) >=window_size:
        context = pre_context[-window_size:] + post_context[:window_size]
    
    elif len(pre_context) < window_size:
        post_index = 2*window_size- len(pre_context)
        context = pre_context + post_context[:post_index]
        
    elif len(pre_context) < window_size:
        pre_index = 2*window_size- len(post_context)
        context = pre_context[-pre_index:] + post_context
    
    else:
        print("Weird condition. Take note")
        context = pre_context + post_context
         
    return context

             
def main():
    
    #Getting the data from the training file
    context_data = getTrainingContextData()
    
    #Adding data from the test file to the same context data
    context_data = getTestContextData(context_data)
    
    #Reading in the word to vector dataframe
    word_vector_subset = readWordToVector()
    
    context_feature_data = makeFeatureVectorForWordInstance(context_data, word_vector_subset)

if __name__ == '__main__':
    main()