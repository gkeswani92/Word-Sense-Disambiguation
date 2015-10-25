'''
Created on Oct 13, 2015

@author: gaurav
'''

from collections         import defaultdict, OrderedDict
from numpy.linalg        import norm
from DataProcessing.Util import initializeXMLParser, dir_path, training_file, readWordToVector, saveContextVectorData, preProcessContextData, test_file, norm_word_counts, gaussian_weighting
from collections import Counter
from scipy.signal import gaussian

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
        
        #break;#TODO: Remove this breakpoint. Only testing for one word type right now
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

def makeFeatureVectorForWordInstance(context_data, word_vector_subset, word_freqs, window_size = 1000, std=10):
    '''
        Creates the feature vector for each word instance by reading the word vectors
        from the word to vec data frame that we created
    '''
    for word_type, word_type_data in context_data.iteritems():
        for data_type, instance_details in word_type_data.iteritems():
            for instance, context_details in instance_details.iteritems():

                # Always grab the entire context; simply choose the gaussian sd differently
                context          = getContextWordsinWindow(context_details, window_size)
                gaussian_weights = getGaussianWeights(context_details, std)
                feature_vector   = createFeatureVectorFromContext(context, word_vector_subset, word_freqs, gaussian_weights)
                context_data[word_type][data_type][instance].update({"Feature_Vector":feature_vector})
                
    return context_data

def getGaussianWeights(context_details, std):
    context_len = len(context_details['Pre-Context']) + len(context_details['Post-Context'])
    if gaussian_weighting == True:
        num_weights = 2 * max(len(context_details['Pre-Context']), len(context_details['Post-Context']))
        weights = gaussian(num_weights, std)
        if len(context_details['Pre-Context']) > len(context_details['Post-Context']):
            trimmed_weights = weights[:context_len]
        elif len(context_details['Pre-Context']) < len(context_details['Post-Context']):
            trimmed_weights = weights[-context_len:]
        else:
            trimmed_weights = weights
        return trimmed_weights
    else:
        return [1.0]*context_len

def createFeatureVectorFromContext(context, word_vector_subset, word_freqs, gaussian_weights):
    '''
        Creates the feature vector from the google word2vec vectors depending on 
        the context words passed in
    '''
    token_vectors           = word_vector_subset.ix[:,context]
   
    # Before summing, weight the vectors by their inverse counts
    counts_weights              = [1.0/word_freqs[word] if word in word_freqs else 1.0 for word in token_vectors.columns]
    weighted_token_vectors  = token_vectors * counts_weights * gaussian_weights
    vector_sum              = weighted_token_vectors.sum(axis = 1)
    normalised_vector       = vector_sum / norm(vector_sum)
    normalised_vector_list  = normalised_vector.tolist()
    return normalised_vector_list
                  
def getContextWordsinWindow(context_details, window_size):
    '''
        Gets the appropriate context words from the pre context and the post 
        context depending on the window size that is passed in
    '''
    pre_context  = context_details['Pre-Context']
    post_context = context_details['Post-Context']    
    context      = []
    
    if len(pre_context) >= window_size and len(post_context) >= window_size:
        context = pre_context[-window_size:] + post_context[:window_size]
    
    elif len(pre_context) < window_size and len(post_context) >= window_size:
        post_index = 2*window_size- len(pre_context)
        context = pre_context + post_context[:post_index]
        
    elif len(post_context) < window_size and len(pre_context) >= window_size:
        pre_index = 2*window_size- len(post_context)
        context = pre_context[-pre_index:] + post_context
    
    else:
        context = pre_context + post_context
         
    return context


def getWordFreqs(context_data):
    '''
        Loop over all word in all contexts, and acquire counts for each of the words.
        Use these counts to downweight the weight vectors while they're being summed.

        When running the test set, if a word count is not found, consider the count to
        have been = 1, since a word which occurs in the test set but not the training
        set is likely to be rare, and thus have count roughly = 1
    '''
    word_counter = Counter()
    for _, word_type_data in context_data.iteritems():
        for _, context_details in word_type_data['training'].iteritems():
            word_counter.update(context_details['Pre-Context'])
            word_counter.update(context_details['Post-Context'])

    return word_counter

def main():
    
    #Getting the data from the training file
    context_data = getTrainingContextData()
    print("Retrieved data from the training file")

    #Adding data from the test file to the same context data
    context_data = getTestContextData(context_data)
    print("Retrieved data from the test file")

    #Obtain training word counts to be used as (inverse) vector weights
    if norm_word_counts == True:
        word_freqs = getWordFreqs(context_data)
    else:
        word_freqs = Counter()

    #Reading in the word to vector dataframe
    word_vector_subset = readWordToVector()
    print("Retrieved the word2vec dataset")
    
    #Create the feature vector for each instance id in the above data structure and save it in JSON format
    context_feature_data = makeFeatureVectorForWordInstance(context_data, word_vector_subset, word_freqs)
    saveContextVectorData(context_feature_data)
    print("Created the word vectors for all word types and their instances")

if __name__ == '__main__':
    main()