"""
Level	Method	Industry Use	Pros	Cons
1️⃣	One-Hot Encoding	Basic NLP, simple models	Easy to implement	Sparse, inefficient
2️⃣	Indexer (Feature Map)	Bag-of-Words, ML models	Smaller representation	No frequency info
3️⃣	Count Vectors (Bag-of-Words)	Classic ML models	Captures word frequency	Ignores importance
4️⃣	TF-IDF	Text classification	Weighs important words	Ignores word order
5️⃣	Word Embeddings (Word2Vec, GloVe, FastText)	Deep Learning, Chatbots	Captures meaning	Fixed-size vectors
6️⃣	Contextual Embeddings (BERT, GPT)	State-of-the-art NLP	Best for context & meaning	Computationally expensive

"""

# Methods

# Bag-of-Word Feature Extraction Methods

## One-Hot Encoding (0/1 Features) 

> ### Level: 1
> ### Usage
> * Basic NLP
> * Simple models like perceptron
> ### Pros:
> * Easy to implement
> ### Cons
> * Sparse
> * Inefficient


## Indexer (Feature Map)

> ### Level: 2
> ### Usage
> * Bag-of-Words
> * ML models
> ### Pros:
> * Smaller representation
> ### Cons
> * No frequency info
> * Ignores importance of words

## N-Gram 

> ### Level: 3
> ### Usage
> * Bag-of-Words
> ### Pros:
> * Captures frequency of a word, n words in order
> ### Cons
> * Ignores the importance of words

## TF-IDF

> ### Level: 4
> ### Usage
> * Bag-of-Words
> * Text classification
> ### Pros:
> * Weighs importance of words
> ### Cons
> * Ignores word order

# Word Embeddings