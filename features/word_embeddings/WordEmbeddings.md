
5️⃣	Word Embeddings (Word2Vec, GloVe, FastText)
Deep Learning, 
Chatbots	Captures meaning	
Fixed-size vectors

# Word Embeddings

## Word2Vec

> ### Cons
> 1) Each word has a single fixed vector representation regalrdless of its meaning
>   1) Example: "bank" in "river bank" gets the same embedding as "financial bank"
> 2) Vocabulary Limitation
>    1) Only learns mebedding for words seen in the training corpus
>    2) New or rare words (Out-of-Vocabulary, OOV) are not represented
>    3) Cannot handle subword information (different forms of the same word)
> 3) Requires a large corpus for good performance
> 4) No knowledge of syntax or hierachical Structure
>   1) leanrs relationships based on co-occurence but does not capture grammar or word order
>   2) Sentences with difference structure but the sma words have similar embeddings
> 5) Cannot handle Polysemy
>   1) since each word gets only one vector, polysemous words (word with multiple meansings ) are problematic
> 6) Hard to interpret
>    1) unlike models with decision trees, its not easy to explain why a particular word has a certain position in the vector space
>    2) learned embeddings are dense vectors, making them difcult to interpret directly
> 7) No Sentence level or document level representation
>    1) Word at the word level only
>    2) Additional methods like averaging word vectors (which loses information) or using more advanced models like transfomers are needed
> ### How it Works
> 1) Build Vocabulary: Create two mappings to convert words into numerical indices:
>       > Example: word_to_index = {"sky": 0, "blue": 1, "is": 2, "bright": 3}
         index_to_word = {0: "sky", 1: "blue", 2: "is", 3: "bright"}
>    
> 2)  Initialize Two Weight Matrices (W1 & W2)
>    1) These weights are randomly initialized and will be trained
>    2) W1 stores word embeddings, while W2 is only used during training to get scores for each word.
> 3) Generate Training Data
>    1)  CBOW → Context words → Predict target word
>    2) Skip-gram → Target word → Predict context words
> 4) Forward Pass
>    1) Convert words → indices
>    2) Look up word vectors in W1
>    3) Compute hidden layer (average of context vectors)
>    4) Multiply hidden layer by W2 to get scores for each word
>    5) Apply softmax to get probabilities
>     > h = average(W1[context_indices].axis(0)) <br>
        u = W2.T @ h <br>
        y_pred = softmax(u)  
  
