# models.py

import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence



#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """

        # vocab_index = {

        #     'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 
        #     'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 
        #     't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, ' ': 26
        
        # }

        # # Convert context to tensor of indices and add batch dimension
        # context_indices = [vocab_index[c] for c in context if c in vocab_index]
        # context_tensor = torch.tensor(context_indices).unsqueeze(0).long()  # Shape: (1, seq_len)
        
        # # Get the model's output and make prediction
        # with torch.no_grad():
        #     output = self.forward(context_tensor)
        #     prediction = torch.argmax(output, dim=1).item()  # Take index with highest score
        # return prediction
        print("Something.........")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier, nn.Module):


    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)  # Using GRU
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)      # GRU returns only the hidden state
        logits = self.fc(hidden[-1])        # Apply fully connected layer to last hidden state
        output = F.softmax(logits, dim=1)   # Apply Softmax to get probabilities
        return output

        

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)







def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """

    vocab_size = 27
    embed_size = 32  # small embedding size for this task
    hidden_size = 32  # small hidden size, adjustable
    output_size = 2  # two classes: consonant and vowel
    epochs     = 10



    model = RNNClassifier(vocab_size, embed_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Prepare training data
    train_data = [(ex, 0) for ex in train_cons_exs] + [(ex, 1) for ex in train_vowel_exs]
    dev_data = [(ex, 0) for ex in dev_cons_exs] + [(ex, 1) for ex in dev_vowel_exs]

    def string_to_tensor(s):
        indices = [vocab_index.index_of(c) for c in s]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for ex, label in train_data:
            optimizer.zero_grad()
            input_tensor = string_to_tensor(ex)
            target = torch.tensor([label], dtype=torch.long)
            output = model(input_tensor)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on development data
        model.eval()
        correct = 0
        with torch.no_grad():
            for ex, label in dev_data:
                input_tensor = string_to_tensor(ex)
                target = torch.tensor([label], dtype=torch.long)
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                correct += (pred == target).sum().item()

        accuracy = correct / len(dev_data)
        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}, Dev Accuracy: {accuracy:.4f}')

    return model



#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")