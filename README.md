# NLP-SMSSpamCollection
SMS Spam Collection

## Introduction
Today, internet and social media have become the fastest and easiest ways to get information. In this age, reviews, opinions, feedbacks, messages and recommendations have become significant source of information. Thanks to advancement in technologies, we are now able to extract meaningful information from such data using various Natural Language Processing (NLP) techniques. NLP , a branch of Artificial Intelligence (AI), makes use of computers and human natural language to output valuable information. NLP is commonly used in text classification task such as spam detection and sentiment analysis, text generation, language translations and document classification.

## Purpose
Build SMS Spam Classification Model using Naive Bayes and Random forest.

## Data set
The SMS (text) data was downloaded from UCI datasets. It contains 5,574 SMS phone messages. The data were collected for the purpose of mobile phone spam research and have already been labeled as either spam or ham.<br />
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

## Data Cleaning 
 * Dropping special Characters
 * Converting to Small case
 * Dropping Stopwords
 * Stemming or Lemmatization
 * Word Embedding - Vectorization
   * Bag of words
   * TF-IDF

### **Stemming and Lemmatization** are text normalization techniques within the field of Natural language Processing that are used to prepare text, words, and documents for further processing. 

## Stemming
Stemming is the process of producing morphological variants of a root/base word. Stemming programs are commonly referred to as stemming algorithms or stemmers.
Often when searching text for a certain keyword, it helps if the search returns variations of the word. For instance, searching for “boat” might also return “boats” and “boating”. Here, “boat” would be the stem for [boat, boater, boating, boats]. <br />
Stemming is a somewhat crude method for cataloging related words; it essentially chops off letters from the end until the stem is reached. This works fairly well in most cases, but unfortunately English has many exceptions where a more sophisticated process is required. In fact, spaCy doesn’t include a stemmer, opting instead to rely entirely on lemmatization.

### Porter Stemmer
One of the most common — and effective — stemming tools is Porter’s Algorithm developed by Martin Porter in 1980. The algorithm employs five phases of word reduction, each with its own set of mapping rules.<br />
```
from nltk.stem.porter import PorterStemmer
```

### Snowball Stemmer
This is somewhat of a misnomer, as Snowball is the name of a stemming language developed by Martin Porter. The algorithm used here is more accurately called the “English Stemmer” or “Porter2 Stemmer”. It offers a slight improvement over the original Porter stemmer, both in logic and speed.<br />
```
from nltk.stem.snowball import SnowballStemmer
```


## Lemmatization
In contrast to stemming, lemmatization looks beyond word reduction and considers a language’s full vocabulary to apply a morphological analysis to words. The lemma of ‘was’ is ‘be’ and the lemma of ‘mice’ is ‘mouse’. Lemmatization is typically seen as much more informative than simple stemming, which is why Spacy has opted to only have Lemmatization available instead of Stemming Lemmatization looks at surrounding text to determine a given word’s part of speech, it does not categorize phrases. <br /><br />

One thing to note about lemmatization is that it is harder to create a lemmatizer in a new language than it is a stemming algorithm because we require a lot more knowledge about structure of a language in lemmatizers. <br />
Stemming and Lemmatization both generate the foundation sort of the inflected words and therefore the only difference is that stem may not be an actual word whereas, lemma is an actual language word. <br />
Stemming follows an algorithm with steps to perform on the words which makes it faster. Whereas, in lemmatization, you used a corpus also to supply lemma which makes it slower than stemming. you furthermore might had to define a parts-of-speech to get the proper lemma. <br />
The above points show that if speed is concentrated then stemming should be used since lemmatizers scan a corpus which consumes time and processing. It depends on the problem you’re working on that decides if stemmers should be used or lemmatizers.


## Bag of word
A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling, such as with machine learning algorithms.<br />

The approach is very simple and flexible, and can be used in a myriad of ways for extracting features from documents.<br />

A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:
 * A vocabulary of known words.
 * A measure of the presence of known words.<br />

It is called a “bag” of words, because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document.

## Drawbacks of using a Bag-of-Words (BoW) Model
We start facing issues when we come across new sentences:
 * If the new sentences contain new words, then our vocabulary size would increase and thereby, the length of the vectors would increase too.
 * Additionally, the vectors would also contain many 0s, thereby resulting in a sparse matrix (which is what we would like to avoid)
 * We are retaining no information on the grammar of the sentences nor on the ordering of the words in the text.

## TF-IDF
A problem with scoring word frequency is that highly frequent words start to dominate in the document (e.g. larger score), but may not contain as much “informational content” to the model as rarer but perhaps domain specific words.<br />

One approach is to rescale the frequency of words by how often they appear in all documents, so that the scores for frequent words like “the” that are also frequent across all documents are penalized.<br />

This approach to scoring is called Term Frequency – Inverse Document Frequency, or TF-IDF for short, where:

**Term Frequency**: is a scoring of the frequency of the word in the current document.<br />
**Inverse Document Frequency**: is a scoring of how rare the word is across documents.<br />
The scores are a weighting where not all words are equally as important or interesting.<br />

The scores have the effect of highlighting words that are distinct (contain useful information) in a given document.
<br /><br />
## Naive Bayes

## Random Forest

## Model results 
| Sno  | Normalization | Vectorization |   Algortihm   |   Accuracy   | Precision | Recall | fscore |
| ---- | ------------- | ------------- | ------------- | ------------ | --------- | ------ | ------ |
|   1	 | Stemming	     | Bag of word	 | Naive Bayes	 | 98.47533632  |    0.944  |  0.95  |  0.947 |
|   2	 | Lemmatization | Bag of word	 | Naive Bayes	 | 98.20627803  |    0.932  |  0.944 |  0.938 |
|   3	 | Stemming	     | TF-IDF	       | Naive Bayes	 | 96.95067265  |    1.0    |  0.788 |  0.881 |
|   4	 | Lemmatization | TF-IDF	       | Naive Bayes	 | 97.21973094  |    1.0    |  0.806 |  0.893 |
|   5	 | Stemming	     | Bag of word	 | Random Forest | 98.11659193  |    0.993  |  0.875 |  0.93  |
|   6	 | Lemmatization | Bag of word	 | Random Forest | 98.29596413  |    1.0    |  0.881 |  0.937 |
|   7	 | Stemming	     | TF-IDF	       | Random Forest | 98.38565022  |    1.0    |  0.888 |  0.94  |
|   8	 | Lemmatization | TF-IDF	       | Random Forest | 98.20627803  |    1.0    |  0.875 |  0.933 |

## Precision

## Recall

## fscore

## CONCLUSION

### References
https://towardsdatascience.com/stemming-vs-lemmatization-2daddabcb221#:~:text=Stemming%20and%20Lemmatization%20both%20generate,words%20which%20makes%20it%20faster <br />
https://towardsdatascience.com/nlp-spam-detection-in-sms-text-data-using-deep-learning-b8632db85cc8 <br />
https://towardsdatascience.com/build-sms-spam-classification-model-using-naive-bayes-random-forest-43465d6617ed <br />
https://machinelearningmastery.com/gentle-introduction-bag-words-model/ <br />
https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/ <br />
 
