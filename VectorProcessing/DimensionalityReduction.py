'''
Created on Oct 17, 2015

@author: gaurav
'''

from DataProcessing.Util import readContextVectorData, savePredictionsToCSV
from collections import OrderedDict, defaultdict
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
import numpy

def performDimensionalityReduction(context_vector):
    '''
        Applies TSNE on the feature vector of each of the word instances and creates
        one model for each word type
    '''
    feature_vector_data = defaultdict(dict)
    word_type_model     = {}
    
    for word_type, word_type_data in context_vector.iteritems():
        print("Currently working on : {0}".format(word_type))
        feature_vector_word_type = OrderedDict()
        
        #Reading in all the feature vectors for the given word type
        for _, instance_details in word_type_data.iteritems():
            for instance, context_details in instance_details.iteritems():
                
                #Training data with have the sense id's while test data will have ['<UNKNOWN>']
                senses = context_details.get('Sense',['<UNKNOWN>'])
                for sense in senses:
                    feature_vector_word_type[(instance, sense)] = context_details["Feature_Vector"]
        
        #Applying TSNE on all the feature vectors
        feature_vector_array = numpy.array(feature_vector_word_type.values())
        model = TSNE(n_components=2, random_state=0, perplexity=5, metric="cosine")
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
            
            #Need to grab the TSNE vectors and senses of only the training data
            #Thus, we ignore all the test data since we have marked that with an
            #<UNKNOWN> sense
            if  sense != "<UNKNOWN>":
                vectors.append(instance_sense_dict.values()[i])
                senses.append(sense)
            
        vectors = numpy.array(vectors)
        senses = numpy.array(senses)
        nb = GaussianNB()
        nb.fit(vectors, senses)
        naive_bayes_model_word_type[word_type] = nb
    
    return naive_bayes_model_word_type

def predictSenseOfTestData(naive_bayes_model, feature_vector_data):
    '''
        Uses the naive bayes model created using the training data to predict
        the senses of the test data
    '''
    test_predictions = []
    
    for word_type, instance_sense_dict in feature_vector_data.iteritems():
        nb = naive_bayes_model[word_type]
        
        for instance_sense, feature in instance_sense_dict.iteritems():
            if instance_sense[1] == '<UNKNOWN>':
                test_predictions.append([instance_sense[0]] + list(nb.predict([feature])))
    
    return test_predictions
                
def plotModel():
    pass
#     #print([ord(x[1][-1]) for x in feature_vector_word_type])
#     sense_id_list = [x[1] for x in feature_vector_word_type]
#     unique_sense_id_list = list(set(sense_id_list))
#     sense_id_colors = [unique_sense_id_list.index(x) for x in sense_id_list]
#          
#     #plt.scatter([x[0] for x in model.embedding_], [x[1] for x in model.embedding_], c=[ord(x[1][-1])%6 for x in feature_vector_word_type])
#     plt.scatter([x[0] for x in model.embedding_], [x[1] for x in model.embedding_], c=sense_id_colors)
#     plt.show()
#     #print(model.embedding_)
#     #break
                

def main():
    
    #Reads in the json file for the context vector data
    context_vector = readContextVectorData()
    
    feature_vector_data, _ = performDimensionalityReduction(context_vector)
    print("Reduced the dimensionality of the training and test vectors")
    
    naive_bayes_model = createNaiveBayesModel(feature_vector_data)
    print("Created the naive bayes model using the training vectors")
    
    test_predictions = predictSenseOfTestData(naive_bayes_model, feature_vector_data)
    print("Predicted the sense of all test instances")
    savePredictionsToCSV(test_predictions)    

if __name__ == "__main__":
    main()
    