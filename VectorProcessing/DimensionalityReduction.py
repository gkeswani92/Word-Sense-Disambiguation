'''
Created on Oct 17, 2015

@author: gaurav
'''

from DataProcessing.Util import readContextVectorData
from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy

def performDimensionalityReduction(context_vector):
    
    for word_type, word_type_data in context_vector.iteritems():
        print(word_type)
        feature_vector = OrderedDict()
        
        #Reading in all the feature vectors for the given word type
        for _, instance_details in word_type_data.iteritems():
            for instance, context_details in instance_details.iteritems():
                senses = context_details['Sense']
                if len(senses) == 1:
                    feature_vector[(instance, senses[0])] = context_details["Feature_Vector"]
                
        feature_vector_array = numpy.array(feature_vector.values())
        model = TSNE(n_components=2, random_state=0, perplexity=5, metric="cosine")
        model.fit(feature_vector_array)
        #print([ord(x[1][-1]) for x in feature_vector])
        sense_id_list = [x[1] for x in feature_vector]
        unique_sense_id_list = list(set(sense_id_list))
        sense_id_colors = [unique_sense_id_list.index(x) for x in sense_id_list]
            
        #plt.scatter([x[0] for x in model.embedding_], [x[1] for x in model.embedding_], c=[ord(x[1][-1])%6 for x in feature_vector])
        plt.scatter([x[0] for x in model.embedding_], [x[1] for x in model.embedding_], c=sense_id_colors)
        plt.show()
        #print(model.embedding_)
        #break
                

def main():
    
    #Reads in the json file for the context vector data
    context_vector = readContextVectorData()
    
    performDimensionalityReduction(context_vector)
    
if __name__ == "__main__":
    main()
    