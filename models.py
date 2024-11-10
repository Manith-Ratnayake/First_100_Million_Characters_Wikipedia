# models.py

import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """

        vocab_index = {

            'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 
            'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 
            't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, ' ': 26
        
        }

        # Convert context to tensor of indices and add batch dimension
        context_indices = [vocab_index[c] for c in context if c in vocab_index]
        context_tensor = torch.tensor(context_indices).unsqueeze(0).long()  # Shape: (1, seq_len)
        
        # Get the model's output and make prediction
        with torch.no_grad():
            output = self.forward(context_tensor)
            prediction = torch.argmax(output, dim=1).item()  # Take index with highest score
        return prediction



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


    def __init__(self, embed_size, hidden_size, input_size=27, output_size=2):
        super(RNNClassifier, self).__init__()
        self.embedding  = nn.Embedding(input_size, embed_size)
        self.gru        = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc         = nn.Linear(hidden_size, output_size)

    def forward(self, context_tensor):       
        embedded    = self.embedding(context_tensor)  # Shape: (batch_size, seq_len, embed_size)   # Convert context indices to embedding
        _ , hidden  = self.gru(embedded)            # `hidden` is of shape (1, batch_size, hidden_size)         # Pass through GRU layer
        output      = self.fc(hidden.squeeze(0))        # Shape: (batch_size, output_size)   # Fully connected layer on hidden state
        return output

        

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)



def string_to_tensor():
    pass


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

    # Hyperparameters
    embed_size = 64      # Size of the embedding vectors
    hidden_size = 64     # Number of hidden units in GRU
    output_size = 2      # Binary output: consonant (0) or vowel (1)
    learning_rate = 0.0001
    num_epochs = 10

    # Instantiate model, loss function, and optimizer
    model = RNNClassifier( embed_size = embed_size, hidden_size = hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    vocab_index = {
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 
        'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 
        't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, ' ': 26
    }


    # Prepare training data as a list of (sequence, label) pairs
    train_data = [(ex, 0) for ex in train_cons_exs] + [(ex, 1) for ex in train_vowel_exs]
    dev_data = [(ex, 0) for ex in dev_cons_exs] + [(ex, 1) for ex in dev_vowel_exs]


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for context, label in train_data:
            context_indices = [vocab_index[c] for c in context if c in vocab_index] # Convert context to indices and label to tensor
            context_tensor = torch.tensor(context_indices).unsqueeze(0).long()      # Shape: (1, seq_len)
            label_tensor = torch.tensor([label])
           
            optimizer.zero_grad()  # Zero gradients, perform forward pass, compute loss, backpropagate, and update weights
            output = model(context_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}")

   
    model.eval()  # Evaluation on dev data
    correct = 0
    with torch.no_grad():
        for context, label in dev_data:
            if model.predict(context) == label:
                correct += 1
    accuracy = correct / len(dev_data) * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")

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