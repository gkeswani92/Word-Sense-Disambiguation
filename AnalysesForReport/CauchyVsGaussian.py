__author__ = 'Jonathan Simon'

import numpy as np
import matplotlib.pyplot as plt

def getGaussian(x):
    '''
    Unnormalized standard normal pdf
    '''
    return np.exp(-x**2 / 2.0)

def getCauchy(x):
    '''
    Unnormalized Cauchy pdf
    '''
    return 1.0 / (1 + x**2)

X = np.arange(-5,5.02,.02)
Y_Gaussian = map(getGaussian, X)
Y_Cauchy = map(getCauchy, X)

plt.plot(X, Y_Gaussian, linewidth=2, label='Gaussian: ' + r'$\mathrm{e}^{-\frac{x^2}{2}}$')
plt.plot(X, Y_Cauchy, linewidth=2, label='Cauchy: ' + r'$\frac{1}{1+x^2}$')
plt.legend()
plt.axis([-5, 5, -.1, 1.15])

fig = plt.gcf()
fig.set_size_inches(8, 5)

dir_name = '/Users/Macbook/Documents/Cornell/CS 4740 - Natural Language Processing/Project 2/Word-Sense-Disambiguation/AnalysesForReport/'
fig.savefig(dir_name+'Gaussian_vs_Cauchy.png', dpi=150)