# models.py

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections

import time
import matplotlib.pyplot as plt



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


    def __init__(self, vocabulary_size, embedding_size, hidden_size, output_size, context_length, device):
        
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.context_length = context_length
        self.device = device


    def forward(self, x):

        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)  
        hidden = hidden[-1]  
        logits = self.fc(hidden)
        output = F.softmax(logits, dim=1)
        return output 


    def predict(self, context):

        context = context[0:self.context_length]
        
        self.eval()
        with torch.no_grad():
            context           = StringToTensor(context).to(self.device)
            output            = self(context)
            prediction        = torch.argmax(output, dim=1).item()
            return prediction


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def StringToTensor(sentence : str):
   
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = {char: i  for i, char in enumerate(vocab)}

    indices = [vocab_index[c] for c in sentence]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


def CharacterIndex(character : str) -> int:
   
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = {char: i for i, char in enumerate(vocab)}
    return vocab_index.get(character, -1)  


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
    vocabulary_size = len(vocab_index)
    output_size     = 2 

    embedding_size   = int(input("Enter embedding size : "))
    hidden_size      = int(input("Enter hidden size : "))
    epochs           = int(input("Enter epochs : "))
    context_length   = int(input("Enter context_length : "))

    learning_rate    = 0.0005

    best_loss             = float('inf')
    patience_counter      = 0
    patience_limit        = 5
    patience_delta        = 0.000001
    average_training_loss = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = RNNClassifier(vocabulary_size, embedding_size, hidden_size, output_size, context_length, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_sentences = len(train_cons_exs) + len(train_vowel_exs)
    start_time = time.time()

    for epoch in range(epochs):

        model.train()
        total_loss = 0 

        for consonant_example, vowel_example in zip(train_cons_exs, train_vowel_exs):
            
            optimizer.zero_grad()
            input_tensor            = StringToTensor(consonant_example).to(device)
            target                  = torch.tensor([0], dtype=torch.long).to(device)
            output                  = model(input_tensor)
            loss                    = criterion(output, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()        

            optimizer.zero_grad()
            input_tensor            = StringToTensor(vowel_example).to(device)
            target                  = torch.tensor([1], dtype=torch.long).to(device)
            output                  = model(input_tensor)
            loss                    = criterion(output, target)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()        
 
            #print(f"Input tensor : {input_tensor} \nTarget : {target} \nOutput : {output} \nLoss : {loss}")
               
        average_training_loss = total_loss / total_sentences
        print(f"Epoch : {epoch + 1} \nTotal Loss : {total_loss} \nAverage Training Loss : {average_training_loss}\n\n ")

        if  best_loss - average_training_loss > patience_delta:
            best_loss = average_training_loss
            patience_counter = 0
            best_model = model.state_dict()  
            
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print("Early stopping triggered")   
            break

    model.load_state_dict(best_model)    
    end_time = time.time()

    print("Average Training Loss : ", average_training_loss)
    print("Embdedding size : ", embedding_size)
    print("Hidden size : ", hidden_size)
    print("Epochs : ", epochs)
    print("Learning Rate : ", learning_rate)
    print(f"Time taken for model training is {end_time - start_time:.2f} seconds")
    print("Average Training Loss : ", average_training_loss)
    
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
    def _init_(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

class RNNLanguageModel(LanguageModel, nn.Module):

    def __init__(self, vocabulary_size, embedding_size, hidden_size, output_size, device):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
        self.hidden_size = hidden_size

    def forward(self, x, hidden_state=None):
        # Only initialize hidden_state once when it's None (first forward pass of a sequence)
        if hidden_state is None:
            hidden_state = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state to zeros (size: (1, batch_size, hidden_size))

        embed = self.embedding(x)
        rnn_output, hidden_state = self.gru(embed, hidden_state)  # Pass through GRU
        logits = self.fc(rnn_output)  # Output layer

        return logits, hidden_state

    def get_log_prob_single(self, next_char, context):
        self.eval()
        with torch.no_grad():
            
            context_tensor = StringToTensor(context).to(self.device)  # Converts string to tensor of indices
            embedded_context = self.embedding(context_tensor)  # Convert indices to embeddings
            rnn_output, _ = self.gru(embedded_context)  # Embedded context pass to GRU for output
            logits = self.fc(rnn_output)  # RNN output to fc

            next_char_index = CharacterIndex(next_char)
            if next_char_index == -1:
                raise ValueError(f"Character '{next_char}' not found in vocabulary.")
            
            # Get the log probabilities for the last timestep in the RNN output
            log_probabilities = torch.log_softmax(logits[0, -1, :], dim=-1)  # Softmax over the vocabulary
            log_probability = log_probabilities[next_char_index].item()  # Get the log probability for the next char
            
        return log_probability

    def get_log_prob_sequence(self, next_chars, context):

        total_log_probability = 0.0

        for char in next_chars: 
            log_probability = self.get_log_prob_single(char, context)
            total_log_probability += log_probability
            context += char  
        
        return total_log_probability
    
    
def Plot_Metrics(train_losses, train_accuracies, dev_losses, dev_accuracies, dev_log_probs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 8))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='red', marker='o')
    plt.plot(epochs, dev_losses, label='Dev Loss', color='orange', marker='x')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs, dev_accuracies, label='Dev Accuracy', color='green', marker='x')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Log Probability
    plt.subplot(2, 2, 3)
    plt.plot(epochs, dev_log_probs, label='Dev Log Probability', color='purple', marker='o')
    plt.title('Log Probability per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Log Probability')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_epoch(model_language, train_chunks, optimizer, criterion, vocab_index, burn_in_length, device):
    model_language.train()
    train_loss, train_accuracy, train_batches = 0, 0, 0
    hidden_state = None

    for chunk in train_chunks:
        burn_in_context = chunk[:burn_in_length]
        prediction_context = chunk[burn_in_length:-1]
        next_chars = chunk[burn_in_length + 1:]

        burn_in_tensor = StringToTensor(burn_in_context).to(device)
        context_tensor = StringToTensor(prediction_context).to(device)
        next_chars_tensor = torch.tensor([vocab_index.index_of(c) for c in next_chars], dtype=torch.long).to(device)

        # Forward pass
        _, hidden_state = model_language(burn_in_tensor, hidden_state)
        hidden_state = hidden_state.detach()  # Detach hidden state to prevent backprop through time

        optimizer.zero_grad()
        output, hidden_state = model_language(context_tensor, hidden_state)
        loss = criterion(output.view(-1, output.size(-1)), next_chars_tensor)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted_indices = torch.max(output, dim=-1)
        correct_predictions = (predicted_indices.view(-1) == next_chars_tensor).sum().item()
        accuracy = correct_predictions / len(next_chars_tensor)

        train_loss += loss.item()
        train_accuracy += accuracy
        train_batches += 1

    avg_train_loss = train_loss / train_batches
    avg_train_accuracy = train_accuracy / train_batches
    return avg_train_loss, avg_train_accuracy

def validate_epoch(model_language, dev_chunks, criterion, vocab_index, burn_in_length, device):
    model_language.eval()
    dev_loss, dev_accuracy, dev_batches = 0, 0, 0
    hidden_state = None

    with torch.no_grad():
        for chunk in dev_chunks:
            burn_in_context = chunk[:burn_in_length]
            prediction_context = chunk[burn_in_length:-1]
            next_chars = chunk[burn_in_length + 1:]

            burn_in_tensor = StringToTensor(burn_in_context).to(device)
            context_tensor = StringToTensor(prediction_context).to(device)
            next_chars_tensor = torch.tensor([vocab_index.index_of(c) for c in next_chars], dtype=torch.long).to(device)

            _, hidden_state = model_language(burn_in_tensor, hidden_state)
            output, _ = model_language(context_tensor, hidden_state)

            loss = criterion(output.view(-1, output.size(-1)), next_chars_tensor)
            _, predicted_indices = torch.max(output, dim=-1)
            correct_predictions = (predicted_indices.view(-1) == next_chars_tensor).sum().item()
            accuracy = correct_predictions / len(next_chars_tensor)

            dev_loss += loss.item()
            dev_accuracy += accuracy
            dev_batches += 1

    avg_dev_loss = dev_loss / dev_batches
    avg_dev_accuracy = dev_accuracy / dev_batches
    return avg_dev_loss, avg_dev_accuracy

def compute_dev_log_prob(model_language, dev_chunks, burn_in_length, device):
    dev_log_prob = 0.0
    for chunk in dev_chunks[:5]:  # You can adjust how many chunks to evaluate
        context = chunk[:burn_in_length]  # Use burn-in context for log prob
        next_chars = chunk[burn_in_length + 1:]
        log_prob = model_language.get_log_prob_sequence(next_chars, context)
        dev_log_prob += log_prob

    avg_dev_log_prob = dev_log_prob / len(dev_chunks[:5])  # Average log prob over the selected chunks
    return avg_dev_log_prob

def compute_total_likelihood(model_language, chunks, burn_in_length, device):
    total_log_prob = 0.0  # To store cumulative log probabilities

    for chunk in chunks:
        context = chunk[:burn_in_length]  # Burn-in context
        next_chars = chunk[burn_in_length + 1:]
        
        # Log probability for the current chunk
        log_prob = model_language.get_log_prob_sequence(next_chars, context)
        total_log_prob += log_prob  # Accumulate log probabilities

    # Convert cumulative log probability to likelihood
    total_likelihood = np.exp(total_log_prob)  # May underflow if log_prob is very negative
    print(f"Total Log Likelihood: {total_log_prob:.6f}")
    print(f"Total Likelihood: {total_likelihood:.6e}")  # Scientific notation for clarity
    return total_log_prob, total_likelihood


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """


    vocabulary_size = len(vocab_index)
    output_size = 27

    train_losses, train_accuracies = [], []
    dev_losses, dev_accuracies = [], []
    dev_log_probs = []

    embedding_size = int(input("Embedding size: "))
    hidden_size = int(input("Hidden size: "))
    epochs = int(input("Epochs: "))

    learning_rate = 0.0005
    context_length = 20
    burn_in_length = 6
    chunk_size = context_length + burn_in_length + 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_language = RNNLanguageModel(vocabulary_size, embedding_size, hidden_size, output_size, device).to(device)

    optimizer = optim.Adam(model_language.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def create_chunks(text):
        return [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1)]

    train_chunks = create_chunks(train_text)
    dev_chunks = create_chunks(dev_text)

    for epoch in range(epochs):

        # Train for one epoch
        avg_train_loss, avg_train_accuracy = train_epoch(model_language, train_chunks, optimizer, criterion, vocab_index, burn_in_length, device)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Validate for the same epoch
        avg_dev_loss, avg_dev_accuracy = validate_epoch(model_language, dev_chunks, criterion, vocab_index, burn_in_length, device)
        dev_losses.append(avg_dev_loss)
        dev_accuracies.append(avg_dev_accuracy)

        # Compute log probabilities for the validation set
        avg_dev_log_prob = compute_dev_log_prob(model_language, dev_chunks, burn_in_length, device)
        dev_log_probs.append(avg_dev_log_prob)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {avg_train_accuracy:.4f}, "
              f"Dev Loss = {avg_dev_loss:.4f}, Dev Acc = {avg_dev_accuracy:.4f}, "
              f"Dev Log Prob = {avg_dev_log_prob:.4f}")


    total_log_prob, total_likelihood = compute_total_likelihood(model_language, train_chunks, burn_in_length, device)    
    print(f"\n\nModel's Total Log Likelihood on Training Set: {total_log_prob:.6f}")
    print(f"Model's Total Likelihood on Training Set: {total_likelihood:.6e}")
    

    dev_log_prob, dev_likelihood = compute_total_likelihood(model_language, dev_chunks, burn_in_length, device)
    print(f"\nModel's Total Log Likelihood on Testing Set: {dev_log_prob:.6f}")
    print(f"Model's Total Likelihood on Testing Set: {dev_likelihood:.6e}")

    Plot_Metrics(train_losses, train_accuracies, dev_losses, dev_accuracies, dev_log_probs)
    return model_language