# RNN

## Why RNN?

> 1) Process text sequentially, they remember previous words when predicting sentiment 
> 2) Maintain a hidden state, which stores information about past words in a sentence
> 3) Work well for longer texts, unlike MLP, which ignore context beyond a fixed window

## RNN in sentiment analysis
> 1) Each word is fed one a time into the netword
> 2) The model remembers previous words through a hidden state
> 3) At the end, the final hidden state is used to predict sentiment


## Model Explanation
> 1) W_xh: (input to hidden ) Captures word meaning
>    1) Computes dot product with each word 
> 2) W_hh: (Hidden to Hidden): Stores past words
>    1) Computes dot product by the total of hh += (dot product(W_xH,word) + dotproduct(W_hh, hh)) iteratively
> 3) W_hy: (Hidden to Output): Maps hidden state to sentiment
> 4) y_pred: (Final output) Uses sigmoid for binary sentiment classification