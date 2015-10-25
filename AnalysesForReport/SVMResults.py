'''
Created on Oct 24, 2015

@author: gaurav
Referred http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html for the visualisation construction
'''
import json
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import OrderedDict

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


validation_path = "/Users/gaurav/Downloads/validation.json"
f = open(validation_path, 'r')
validation_file = json.load(f, object_pairs_hook=OrderedDict)

tuple_values = sorted([(Decimal(key.split(',')[1]), Decimal(key.split(',')[2]), Decimal(values)) for key, values in validation_file.iteritems() if key.split(',')[4] == u' 10)'])

def inplace_unique(seq):
    final = []
    for item in seq:
        if '%.2E' % item not in final:
            final.append('%.2E' %item)
    return final

unique_C_range = inplace_unique([key[0] for key in tuple_values])
unique_gamma_range = inplace_unique([key[1] for key in tuple_values])

scores = np.array([key[2] for key in tuple_values], np.float32).reshape(len(unique_C_range), len(unique_gamma_range))

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap='hot')#, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(unique_gamma_range)), unique_gamma_range, rotation=45)
plt.yticks(np.arange(len(unique_C_range)), unique_C_range)
plt.title('Validation accuracy')
plt.show()