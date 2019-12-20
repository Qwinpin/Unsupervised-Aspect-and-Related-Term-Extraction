import torch
from torch import nn


class AspectMatrix(nn.Module):
    def __init__(self, aspects_number, aspects_dim, aspect_weights=None, bias=False):
        """
        Module of aspect extraction
        Each aspect is a row in weights matrix
        """
        super(AspectMatrix, self).__init__()

        self.aspects_number = aspects_number
        self.aspects_dim = aspects_dim
        self.W = nn.Parameter(torch.Tensor(self.aspects_number, self.aspects_dim))

        if aspect_weights is None:
            self.W.data.uniform_(-1.0, 1.0)
        else:
            self.W.data = torch.tensor(aspect_weights).float()

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.aspects_number))
            self.bias.data.uniform_(0.0, 0.0)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.tensor(x).float()
            except:
                raise TypeError
        return torch.matmul(x, self.W), self.W


class Attention(nn.Module):
    """
    Convolutional multidimensial attention
    """
    def __init__(self, aspects_number, emb_dim, bias=True):
        super(Attention, self).__init__()
        self.aspects_number = aspects_number
        self.emb_dim = emb_dim

        # groups number equal to aspect number allows keeping the number of channel
        # one aspect - one attention channel
        self.conv1_3 = torch.nn.Conv2d(aspects_number, aspects_number, kernel_size=(3, self.emb_dim), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=aspects_number)
        self.conv1_5 = torch.nn.Conv2d(aspects_number, aspects_number, kernel_size=(5, self.emb_dim), stride=(1, 1), padding=(2, 0), dilation=(1, 1), groups=aspects_number)
        self.conv1_7 = torch.nn.Conv2d(aspects_number, aspects_number, kernel_size=(7, self.emb_dim), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=aspects_number)
        self.conv1_1 = torch.nn.Conv2d(aspects_number, aspects_number, kernel_size=(1, self.emb_dim), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=aspects_number)

    def forward(self, x, context):
        # expend dimensions of input data for each aspect
        x = x.unsqueeze(1).repeat(1, self.aspects_number, 1, 1)

        x_1 = self.conv1_1(x)
        x_3 = self.conv1_3(x)
        x_5 = self.conv1_5(x)
        x_7 = self.conv1_7(x)

        x = torch.cat([x_1, x_3, x_5, x_7], -1).mean(-1)
        x = nn.Sigmoid()(x)
        return x


class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()

    def forward(self, x, attention):
        return torch.matmul(attention, x)


class Average(nn.Module):
    def __init__(self):
        super(Average, self).__init__()

    def forward(self, x):
        return x.mean(-2)


def embedding_layer(weights):
    num_embeddings, embedding_dim = weights.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)

    emb_layer.load_state_dict({'weight': torch.tensor(weights)})

    emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class AspectExtractor(nn.Module):
    def __init__(self, weights, asp_weights=None, aspects=10):
        super(AspectExtractor, self).__init__()
        self.embedding, _, embedding_dim = embedding_layer(weights)

        self.aspect_matrix = AspectMatrix(aspects, embedding_dim, asp_weights)
        self.attention = Attention(aspects, embedding_dim)
        self.w_sum = WeightedSum()
        self.w_sum.requires_grad = False
        self.sum = Average()
        self.sum.requires_grad = False
        self.linear = nn.Linear(embedding_dim, aspects)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_n):
        embedding = self.embedding(x)
        sentence = self.sum(embedding)
        attention = self.attention(embedding, sentence)

        attention_sentence = self.w_sum(embedding, attention)
        attention_sentence_general = attention_sentence.mean(1)

        aspect_probability = self.sigmoid(self.linear(attention_sentence_general))

        reconstructed_sentence, aspects_weights = self.aspect_matrix(aspect_probability)

        embedding_neg = self.embedding(x_n)
        negative_sentences = self.sum(embedding_neg)

        return attention, attention_sentence, attention_sentence_general, aspect_probability, reconstructed_sentence, negative_sentences, aspects_weights

    def asp_pred(self, x):
        embedding = self.embedding(x)
        sentence = self.sum(embedding)
        attention = self.attention(embedding, sentence)

        attention_sentence = self.w_sum(embedding, attention)
        attention_sentence_general = attention_sentence.mean(1)
        aspect_probability = self.softmax(self.linear(attention_sentence_general))

        return aspect_probability, attention
