'''
Created on Oct 21, 2015

@author: gaurav
'''

from DataProcessing.Util import initializeXMLParser, dir_path, training_file, preProcessContextData, pprint, readWordToVector, saveValidationData
from collections         import defaultdict, OrderedDict
from numpy.linalg        import norm
from sklearn.manifold    import TSNE
from sklearn.naive_bayes import GaussianNB
import random
import numpy
import pickle


sample_size          = 15
window_size_options  = xrange(10,101,10)
n_components_options = xrange(2,21,2)
perplexity_options   = xrange(5, 31, 5)
naive_bayes_window   = [0.01, 0.05, 0.005]


correct_count = 0
prediction_count = 0

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
            instance     = word_instance.attrib['id']
            senses       = [answer.attrib['senseid'] for answer in word_instance.findall('answer')]
            pre_context  = word_instance.find('context').text.split()
            post_context = word_instance.find('context').find('head').tail.split()
            
            #Pre-processing the pre-context and post context
            pre_context = preProcessContextData(pre_context)
            post_context = preProcessContextData(post_context)
            
            #Dividing the training data into training and validation
            training_data[word_type]['training'][instance] = {"Sense":senses, "Pre-Context":pre_context, "Post-Context":post_context }
        
        #Choosing a random set of training data as the validation data
        training_data[word_type] = createValidationData(training_data[word_type])
        
        #break;
    return training_data

def createValidationData(training_data):
    
    #Grabbing 20 or sample size of random keys from the training data and creating a validation dictionary out of it
    random_index = []
    for _ in range(sample_size*len(training_data['training'])/100):
        random_index.append(random.randint(0, len(training_data['training'])-1))
        
    validation = {}
    for index in random_index:
        key = training_data['training'].keys()[index]
        value = training_data['training'][key]
        validation[key] = value
    
    training_data['training'] = dict([(key,value) for key, value in training_data['training'].iteritems() if key not in validation])
    training_data['validation'] = validation
    
    return training_data

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

def performDimensionalityReduction(context_vector, n_component, perplexity):
    '''
        Applies TSNE on the feature vector of each of the word instances and creates
        one model for each word type
    '''
    feature_vector_data = defaultdict(dict)
    word_type_model     = {}
    
    for word_type, word_type_data in context_vector.iteritems():
        feature_vector_word_type = OrderedDict()
        
        #Reading in all the feature vectors for the given word type
        for data_type, instance_details in word_type_data.iteritems():
            for instance, context_details in instance_details.iteritems():
                
                #Training data with have the sense id's while test data will have ['<UNKNOWN>']
                senses = context_details.get('Sense')
                for sense in senses:
                    feature_vector_word_type[(instance, sense, data_type)] = context_details["Feature_Vector"]
        
        #Applying TSNE on all the feature vectors
        feature_vector_array = numpy.array(feature_vector_word_type.values())
        model = TSNE(n_components=n_component, random_state=0, perplexity=perplexity, metric="cosine")
        model.fit(feature_vector_array)
        
        #Storing the model since it will be needed to fit the test data
        word_type_model[word_type] = model
        
        #Converting to a structure of {WordType: {(instanceID, senseID): FeatureVector ... }}
        #TODO: Check why same instance id with different sense has difference feature vectors
        for i in range(len(feature_vector_word_type)):
            feature_vector_data[word_type][feature_vector_word_type.keys()[i]] = list(model.embedding_[i])

    return feature_vector_data, word_type_model

def createNaiveBayesModel(feature_vector_data):
    '''
        Uses the dimensionally reduced feature vectors of each of the instance, sense id pairs
        to create a naive bayes model
    '''
    naive_bayes_model_word_type = {}
    
    for word_type, instance_sense_dict in feature_vector_data.iteritems():
        vectors = []
        senses  = []
        
        for i in xrange(len(instance_sense_dict)):
            sense = instance_sense_dict.keys()[i][1]
            data_type = instance_sense_dict.keys()[i][2]
            
            #Need to grab the TSNE vectors and senses of only the training data
            #Thus, we ignore all the validation data
            if  data_type == "training":
                vectors.append(instance_sense_dict.values()[i])
                senses.append(sense)
            
        vectors = numpy.array(vectors)
        senses = numpy.array(senses)
        nb = GaussianNB()
        nb.fit(vectors, senses)
        naive_bayes_model_word_type[word_type] = nb
    
    return naive_bayes_model_word_type

def predictSenseOfTestData(naive_bayes_model, feature_vector_data, context_feature_data, nb_range):
    '''
        Uses the naive bayes model created using the training data to predict
        the senses of the test data
    '''
    test_predictions = []
    global correct_count, prediction_count
    correct_count = 0
    prediction_count = 0
    
    for word_type, instance_sense_dict in feature_vector_data.iteritems():
        nb = naive_bayes_model[word_type]
        
        for instance_sense, feature in instance_sense_dict.iteritems():

            if instance_sense[2] == 'validation':
                correct_sense = context_feature_data[word_type]['validation'][instance_sense[0]]['Sense']
                naive_bayes_probabilities = nb.predict_proba([feature])[0]
                predictions = [nb.classes_[i] for i in xrange(len(naive_bayes_probabilities)) if abs(max(naive_bayes_probabilities) - naive_bayes_probabilities[i]) < nb_range]
               
                if(correct_sense == predictions):
                    correct_count += 1
                    prediction_count += 1
                else:
                    prediction_count += 1 
                
    return test_predictions

def controller(context_data, word_vector_subset, window_size, n_component, perplexity, nb_range):
    
    #Create the feature vector for each instance id in the above data structure
    context_feature_data = makeFeatureVectorForWordInstance(context_data, word_vector_subset, window_size)
    
    feature_vector_data, _ = performDimensionalityReduction(context_feature_data, n_component, perplexity)
    #print("Performed dimensionality reduction on the vectors")
    
    naive_bayes_model = createNaiveBayesModel(feature_vector_data)
    #print("Created the naive bayes model using the training vectors")\
    
    _ = predictSenseOfTestData(naive_bayes_model, feature_vector_data, context_feature_data, nb_range)
    #print("Predicted the sense of all test instances")
    
    perc_correct = correct_count * 100.0/prediction_count
    return perc_correct
    
def grid_search():
    
    results = {}
    total = len(window_size_options) * len(n_components_options) * len(perplexity_options) * len(naive_bayes_window)
    print("Total steps: {0}".format(total))
    
    #Getting the data from the training file
    context_data = getTrainingContextData()
    print("Grabbed the context data")
    
    #Reading in the word to vector dataframe
    word_vector_subset = readWordToVector()
    print("Grabbed the word_vector_ubset")
    
    for window_size in window_size_options:
        for n_component in n_components_options:
            for perplexity in perplexity_options:
                for nb_range in naive_bayes_window:
                    results[str((window_size, n_component, perplexity))]= controller(context_data, word_vector_subset, window_size, n_component, perplexity, nb_range)
                    total -= 1
                    print("{0} to go".format(total))
    
    pprint(results)
    saveValidationData(results);
    
if __name__ == "__main__":
    grid_search()