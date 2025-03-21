import numpy as np

class RNNPreTrained:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim    # Word vector size (e.g., 50 from Word2Vec)
        self.hidden_dim = hidden_dim  # Number of hidden units

        # Initialize weight matrices
        self.W_xh = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim))   # Input to hidden
        self.W_hh = np.random.uniform(-0.5, 0.5, (hidden_dim, hidden_dim))  # Hidden to hidden (recurrence)
        self.W_hy = np.random.uniform(-0.5, 0.5, (1, hidden_dim))           # Hidden to output

        # Initialize bias terms
        self.b_h = np.zeros(hidden_dim)  # Bias for hidden state
        self.b_y = np.zeros(1)           # Bias for output

    def forward(self, X):
        """
        Forward pass of the RNN.
        X: List of word vectors (one per word in the sentence).
        """
        h_t = np.zeros(self.hidden_dim)  # Initialize hidden state as zeros

        # Process each word in the sentence
        for x_t in X:
            h_t = np.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_t) + self.b_h)

        logits = np.dot(self.W_hy, h_t) + self.b_y  # Final hidden state → Output
        y_pred = 1 / (1 + np.exp(-logits))  # Sigmoid for binary classification

        return y_pred, h_t  # Return prediction and final hidden state
