__author__ = 'Jonathan Simon'

from lxml import etree

data_path = "/Users/Macbook/Documents/Cornell/CS 4740 - Natural Language Processing/Project 2/Word-Sense-Disambiguation/DataProcessing/DataFiles/"
training_file = 'training-data.data'
test_file = 'test-data.data'

parser = etree.XMLParser(recover=True)

training_tree = etree.parse(data_path+training_file, parser=parser)
training_root = training_tree.getroot()

test_tree = etree.parse(data_path+test_file, parser=parser)
test_root = test_tree.getroot()

aggregate_context = []

for word_type in training_root:
    for word_instance in word_type:
        for answer in word_instance.findall('answer'):
            answer.attrib['senseid']
        aggregate_context.extend(word_instance.find('context').text.split())
        aggregate_context.extend(word_instance.find('context').find('head').tail.split())


for word_type in training_root:
    for word_instance in word_type:
        aggregate_context.extend(word_instance.find('context').text.split())
        aggregate_context.extend(word_instance.find('context').find('head').tail.split())

for word_type in test_root:
    for word_instance in word_type:
        aggregate_context.extend(word_instance.find('context').text.split())
        aggregate_context.extend(word_instance.find('context').find('head').tail.split())


# Extact list of all words from among the word2vec vectors
# Check for the presense of special characters, uppercase words, etc
path_to_word_vectors = '/Users/Macbook/Documents/Data_Analysis_and_ML_Projects/word2vec_Analogical_Chaining/Data/GoogleNews-vectors-negative300.txt'
w2v_words = []
with open(path_to_word_vectors) as infile:
    for line in infile:
        w2v_words.append(line.split()[0])
        if len(w2v_words) % 1e5 == 0:
            print "Finished {0} words out of 3 million".format(len(w2v_words))
    w2v_words = w2v_words[1:] # first line is header
# Took ~2min to run to completion

output_path = '/Users/Macbook/Documents/Data_Analysis_and_ML_Projects/word2vec_Analogical_Chaining/Data/3million_unique_words.txt'
with open(output_path,'w') as outfile:
    outfile.write('\n'.join(w2v_words))

# Read the file back in:
with open(output_path) as infile:
    w2v_words = [word.strip('\n') for word in infile.readlines()]

# all-alpha words: 28.7%
# multi-part named entities: 69.0%
#
# Examples of multi-part entities (mostly names and places):
# New_York
# innate_goodness
# Charlestown_NH
# Karla_Knafel
# Winnacunnet_Cooperative
# MD_Ph.D._Principal_Investigator
# La_Rinconada_Country
# GREENVILLE_SC_Amu_Saaka
# Paul_Oscher
# disable_jailbroken_phones

allalpha = [w for w in w2v_words if w.isalpha()]
allalpha_set = set(allalpha)
missing_lowercase = []
missing_uppercase = []
for w in allalpha_set:
    if len(w) == 1:
        continue
    if w[0].isupper():
        if w[0].lower() + w[1:] not in allalpha_set:
            missing_lowercase.append(w)
    else:
        if w[0].upper() + w[1:] not in allalpha_set:
            missing_uppercase.append(w)

# # Upper not lower
# 613738 --> 71.2% of all-alpha words (overwhelmingly names and other proper nouns)
# Feijo
# Jimenez
# Evason
# Kerans
# CoCos
# Palmilla
# CLNO
# Laneau
# Cepia
# Letterston
# # Lower not Upper
# 71511 --> 8.3% of all-alpha words (much more diverse, but mostly rare words)
# spidery
# eVantage
# reactogenicity
# vultured
# capering
# synopsize
# playcallers
# leetle
# indisposition
# milligauss

# For each word that appears in the context, check if it appears among the word2vec words
missing_words_set = set(aggregate_context) - set(w2v_words)

missing_words_counter = Counter([w for w in aggregate_context if w in missing_words_set])

# # Many missing words are non-alpha:
# 10 (402)
# 20 (302)
# 1988 (293)
# 15 (262)
# 1987 (236)
# 1990 (229)
# 1991 (227)
# 1989 (220)
# 12 (214)
# 1985 (199)
# 100 (187)
# 1986 (187)
# 1980 (186)

# # Or due to alternate spellings
# centre (352)
# colour (324)
# programme (315)
# behaviour (253)
# theatre (246)
# defence (214)
# favour (175)
# grey (139)
# organisation (126)

# For each word that appears in the context, check if it appears among the word2vec words
missing_words = set(aggregate_context) - set(w2v_words)
# Out of 53938 unique words in the contexts, only 6302 fail to appear among the w2v words
# That's ~11.68% of the word types
# If we count the raw occurrences of words, however:
binary_missing_words = [1 if word in missing_words else 0 for word in aggregate_context]
print sum(binary_missing_words)
# Overall, we're missing 318,132 word out of 1,408,969 word
# Therefore we're missing ~22.58% of our total words! That's not good!
# However, if we remove the top-15 most common missing words (e.g. punctuation, meaningless interjections, etc).
# then we're down to missing only 22407 words, which is just ~1.59% !!! WOOOHOOO
# Top-15 missing words, along with their frequencies:
# (',', 66111),
# ('.', 51400),
# ('of', 42890),
# ('to', 35472),
# ('and', 34065),
# ('a', 28314),
# ('-', 10995),
# ("'s", 8477),
# (')', 4143),
# ('(', 4117),
# (':', 3164),
# (';', 2811),
# ('?', 2123),
# ("'", 1028),
# ('!', 615),

# Ok, so the punctuation is still an issue, but first make everything lower-case, and see if that helps
aggregate_context_lower = set([word.lower() for word in aggregate_context])
w2v_set_lower = set([word.lower() for word in w2v_words])
len(w2v_set_lower) # Only 2702806 unique words now...
missing_words_lower = set(aggregate_context_lower) - set(w2v_set_lower)
# Now it's 5335 that fail to appear. Still not great...

# Figure out which missing words were most common:
from collections import Counter
missing_word_counter = Counter([word for word in aggregate_context if word in missing_words])
print missing_word_counter.most_common(100)

# However, a fair number of these are common words, only with odd suffixes.
# Let me try running a "stemmer", and then doing it again
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
aggregate_context_set = set(aggregate_context)
missing_words2 = set(st.stem(word) for word in aggregate_context_set) - set(w2v_words)
# This just made things even worse! Now there are 14872 unknown words out of 25547

w2v_nonalpha = [word for word in w2v_words if (not word.isalpha() and '_' not in word)]
# Down from 3million to just 67531
w2v_nonalpha = [word for word in w2v_words if (not word.isalpha()
                                               and '_' not in word
                                               and '##' not in word
                                               and '==' not in word
                                               and '--' not in word
                                               and '..' not in word)]
# Down from 3million to just 53529


# CONCLUSION:
# Punctuation characters are not present among the word2vec words
# If I can't find a given word among the w2v words, there are 2 options:
# 1) Ignore it
# 2) Search for variants of capitalization, spelling, or stemming
# It makes sense to go with 1, since