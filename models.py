# models.py

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def _init_(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier, nn.Module):

    
    def _init_(self, vocabulary_size, embedding_size, hidden_size, output_size):
        super(RNNClassifier, self)._init_()
        
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=False)  # Using GRU
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)  # GRU returns output and hidden state
        
        # If GRU is bidirectional, hidden will have 2 layers (forward and backward)
        # We need to select the final hidden state from the correct direction
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # GRU returns a tuple (hidden, cell_state), so we only need hidden
        
        # Get the last hidden state (for sequence classification)
        # If bidirectional, we concatenate the forward and backward hidden states
        if self.gru.bidirectional:
            hidden = hidden[-2:].transpose(0, 1).contiguous().view(hidden.size(1), -1)
        else:
            hidden = hidden[-1]
        
        logits = self.fc(hidden)  # Apply the fully connected layer
        output = F.softmax(logits, dim=1)  # Apply Softmax to get probabilities
        return output


    def predict(self, context):
        
        self.eval()

        with torch.no_grad():
            context           = string_to_tensor(context)
            output            = self(context)
            value, prediction = torch.max(output, 1)
            return prediction


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def string_to_tensor(s):

    vocab_index = {
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 
        'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 
        't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, ' ': 26
    }

    indices = [vocab_index[c] for c in s]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


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
    
    vocabulary_size = 27
    embedding_size  = 16  # small embedding size for this task
    hidden_size     = 16  # small hidden size, adjustable
    output_size     = 2  # two classes: consonant and vowel
    epochs          = 50


    model     = RNNClassifier(vocabulary_size, embedding_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Prepare training data
    train_data = [(sentence, 0) for sentence in train_cons_exs] + [(sentence, 1) for sentence in train_vowel_exs]
    dev_data = [(sentence, 0) for sentence in dev_cons_exs] + [(sentence, 1) for sentence in dev_vowel_exs]


    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for example, label in train_data:

            optimizer.zero_grad()
            input_tensor            = string_to_tensor(example)
            target                  = torch.tensor([label], dtype=torch.long)
            output                  = model(input_tensor)
            loss                    = criterion(output, target)
            
            loss.backward()                # Backpropagation
            optimizer.step()
            total_loss += loss.item()

            print(f"input tensor : {input_tensor} \nTarget : {target} \nOutput : {output} \nLoss : {loss}\n\n") 

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()  # Save the best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Load the best model if early stopping was triggered
    model.load_state_dict(best_model)    
    return model







'''
        # Evaluate on development data
        model.eval()
        correct = 0
        with torch.no_grad():
            for ex, label in dev_data:
                input_tensor = string_to_tensor(ex)
                target = torch.tensor([label], dtype=torch.long)
                output = model(input_tensor)

                value, pred = torch.max(output, 1)
                print(f"Output : {output} Value : {value} \n Pred {pred} \n\n")
                

                correct += (pred == target).sum().item()

        accuracy = correct / len(dev_data)
        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}, Dev Accuracy: {accuracy:.4f}')

    return model


''' 


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
    def _init_(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def _init_(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index
        # 
        self.vocab_size = len(vocab_index)
        

    def get_log_prob_single(self, next_char, context):
        
        # Convert context to indices
        context_indices = [self.vocab_index[c] for c in context if c in self.vocab_index]
        context_tensor = torch.tensor(context_indices).unsqueeze(0).long()  # Shape: (1, context_len)

        # Forward pass through the RNN
        with torch.no_grad():
            output = self.model_dec(context_tensor)  # Shape: (1, context_len, vocab_size)
        
        # Extract last output
        last_output = output[0, -1]  # Shape: (vocab_size,)
        
        # Get log probability of next_char
        next_char_idx = self.vocab_index.get(next_char, None)
        if next_char_idx is not None:
            log_prob = torch.log_softmax(last_output, dim=0)[next_char_idx].item()
        else:
            log_prob = -float('inf')  # Return -inf if character is not in vocab
        return log_prob

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