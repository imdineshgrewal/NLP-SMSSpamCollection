# NLP-SMSSpamCollection
SMS Spam Collection

### Use case 

### Data set
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

### Data Cleaning 
 Steps : * Dropping special Characters
         * Converting to Small case
         * Dropping Stopwords
         * Stemming or Lemmatization 
              * Nested bullet
                  * Sub-nested bullet etc
          * Bullet list item 2

### Model results 
| Sno  |     Type      | Vectorization |     Model    |  Accuracy    |
| ---- | ------------- | ------------- | ------------ | ------------ |
|   1	 | Stemming	     | Bag of word	 | Naïve Bayes	| 98.47533632  |
|   2	 | Lemmatization | Bag of word	 | Naïve Bayes	| 98.20627803  |
|   3	 | Stemming	     | TF-IDF	       | Naïve Bayes	| 96.95067265  |
|   4	 | Lemmatization | TF-IDF	       | Naïve Bayes	| 97.21973094  |
