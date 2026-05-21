import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1


def tokenize(text):
    """
    Simple tokenizer for cleaned tweet text.
    """
    return re.findall(r"[a-z']+", str(text).lower())


def build_vocab(texts, max_vocab_size=50000, min_freq=2):
    """
    Build vocabulary from training texts only.
    """
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    most_common_words = [
        word
        for word, freq in counter.most_common(max_vocab_size)
        if freq >= min_freq
    ]

    itos = [PAD_TOKEN, UNK_TOKEN] + most_common_words
    stoi = {word: idx for idx, word in enumerate(itos)}

    return stoi, itos, counter


def encode_text(text, stoi, max_len=50):
    """
    Convert text into sequence of word IDs and return real sequence length.
    """
    tokens = tokenize(text)

    ids = [
        stoi.get(token, UNK_IDX)
        for token in tokens
    ]

    real_length = min(len(ids), max_len)

    if real_length == 0:
        real_length = 1

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))

    return ids, real_length


class TweetDataset(Dataset):
    """
    PyTorch Dataset for tweet sentiment classification.
    """
    def __init__(self, dataframe, stoi, max_len=50):
        self.texts = dataframe["clean_text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_ids, length = encode_text(
            self.texts[idx],
            self.stoi,
            self.max_len
        )

        label = self.labels[idx]

        return {
            "input_ids": torch.tensor(text_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float)
        }


class SentimentRNN(nn.Module):
    """
    Recurrent neural network model for sentiment classification.

    It can work as:
    - LSTM
    - GRU
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        output_dim=1,
        num_layers=1,
        dropout=0.4,
        model_type="lstm",
        bidirectional=True
    ):
        super(SentimentRNN, self).__init__()

        self.model_type = model_type.lower()
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_IDX
        )

        if self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )

        elif self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )

        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")

        direction_factor = 2 if bidirectional else 1

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        if self.model_type == "lstm":
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)

        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            final_hidden = torch.cat(
                (forward_hidden, backward_hidden),
                dim=1
            )
        else:
            final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)

        return logits.squeeze(1)
