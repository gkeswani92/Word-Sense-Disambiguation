'''
Created on Oct 13, 2015

@author: gaurav
'''

from collections         import defaultdict, OrderedDict
from numpy.linalg        import norm
from DataProcessing.Util import initializeXMLParser, dir_path, training_file, readWordToVector, saveContextVectorData, preProcessContextData, test_file

def getTrainingContextData():
    
    training_data = OrderedDict()
    
    #Initialising the xml parser for the training and test set
    training_root = initializeXMLParser(dir_path+training_file) 
    
    #Grabbing one word type at a time
    for word_type_xml in training_root:
        word_type = word_type_xml.attrib['item']
        training_data[word_type] = defaultdict(lambda: defaultdict(dict))
        
        #Grabbing the instance id and its list of senses
        for word_instance in word_type_xml:
            instance = word_instance.attrib['id']
            senses   = [answer.attrib['senseid'] for answer in word_instance.findall('answer')]
            pre_context  = word_instance.find('context').text.split()
            post_context = word_instance.find('context').find('head').tail.split()
            
            #Pre-processing the pre-context and post context
            #TODO: Check why this is reducing the accuracy of the model by 1%
            pre_context = preProcessContextData(pre_context)
            post_context = preProcessContextData(post_context)
            
            training_data[word_type]['training'][instance] = {"Sense":senses, "Pre-Context":pre_context, "Post-Context":post_context }
        
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
            
            pre_context = preProcessContextData(pre_context)
            post_context = preProcessContextData(post_context)
            
            test_data[word_type]['test'][instance] = {"Pre-Context":pre_context, "Post-Context":post_context }
            
        #break#TODO: Remove this breakpoint. Only testing for one word type right now
    return test_data

def makeFeatureVectorForWordInstance(context_data, word_vector_subset, window_size = 10):
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
                
    return context_data            
    
def createFeatureVectorFromContext(context, word_vector_subset):
    '''
        Creates the feature vector from the google word2vec vectors depending on 
        the context words passed in
    '''
    token_vectors           = word_vector_subset.ix[:,context]
    vectors_sum             = token_vectors.sum(axis = 1)
    normalised_vectors      = vectors_sum / norm(vectors_sum)
    normalised_vector_list  = normalised_vectors.tolist()
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
        
    elif len(post_context) < window_size:
        pre_index = 2*window_size- len(post_context)
        context = pre_context[-pre_index:] + post_context
    
    else:
        print("Weird condition. Take note")
        context = pre_context + post_context
         
    return context

             
def main():
    
    #Getting the data from the training file
    context_data = getTrainingContextData()
    print("Retrieved data from the training file")
    
    #Adding data from the test file to the same context data
    context_data = getTestContextData(context_data)
    print("Retrieved data from the test file")
        
    #Reading in the word to vector dataframe
    word_vector_subset = readWordToVector()
    print("Retrieved the word2vec dataset")
    
    #Create the feature vector for each instance id in the above data structure and save it in JSON format
    context_feature_data = makeFeatureVectorForWordInstance(context_data, word_vector_subset)
    #pprint(context_feature_data)
    saveContextVectorData(context_feature_data)
    print("Created the word vectors for all word types and their instances")

if __name__ == '__main__':
    main()