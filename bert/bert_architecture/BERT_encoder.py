from bert.bert_architecture.BERT_encoder_layer import BERTEncoderLayer


class BERTEncoder:
    def __init__(self, num_layers, embed_dim, ff_hidden_dim):
        self.layers = [
            BERTEncoderLayer(embed_dim, ff_hidden_dim)
            for _ in range(num_layers)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
