
# Methods

# Bag-of-Word Feature Extraction Methods

## One-Hot Encoding (0/1 Features) 

> ### Level: 1
> ### Pros:
> * Easy to implement
> ### Cons
> * Sparse
> * Inefficient


## Indexer (Feature Map)

> ### Level: 2
> ### Pros:
> * Smaller representation
> ### Cons
> * No frequency info
> * Ignores word order
> * Ignores importance of words

## N-Gram 

> ### Level: 3
> ### Pros:
> * Captures frequency of a word, n words in order
> ### Cons
> * Ignores the importance of words

## TF-IDF

> ### Level: 4
> ### Pros:
> * Weighs importance of words
> ### Cons
> * Ignores word order