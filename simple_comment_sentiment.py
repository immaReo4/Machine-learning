import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)', 'beautiful', 'very good', 'emotional', 'inspiring', 'humanizing' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(', 'waste','futile','exagerrated','exagerrating','sad','depressing','destructive' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
 
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

#training part 
train_set = negative_features + positive_features + neutral_features

 #Setting classsifier part
classifier = NaiveBayesClassifier.train(train_set) 
 
# Predicting part
neg = 0
pos = 0
sentence1= "Awesome movie, I liked it"
sentence1= sentence1.lower()
words = sentence1.split(' ')
for word in words:
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
 
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))


neg2 = 0
pos2 = 0
sentence2= "Terrible movie, but it was good story with a futile ending."
sentence2= sentence2.lower()
words = sentence2.split(' ')
for word in words:
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg2 = neg2 + 1
    if classResult == 'pos':
        pos2 = pos2 + 1
 
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))