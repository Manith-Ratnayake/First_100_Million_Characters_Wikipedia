## 📝 Dataset

This dataset, curated by Matt Mahoney, is derived from the first 100 million characters of Wikipedia. It includes only 27 character types: lowercase letters and spaces. Special characters are replaced with a single space, and numbers are written out as individual digits (e.g., "20" becomes "two zero"). A larger version of this dataset, consisting of 90 million characters for training, 5 million for development, and 5 million for testing, was utilized in Mikolov et al. (2012).

## 📦 Project Folder Setup

- lm_classifier.py - This file serves as the main driver for Part 1. It invokes the train_rnn_classifier function, which trains an RNN classifier on the classification dataset.
- lm.py - This file is the driver for Part 2, and it calls train_lm to train a language model on raw text data.
- models.py - This file contains the skeleton code where you will implement the models and define their training procedures.
- utils.py - This file includes all the necessary utility functions for the project framework.


## 🚀 Project Aim

To develop models for the 2 questions the provided questions

- Question 1 -  RNNs for Classification
- Question 2 -  Implementing a Language Model 


## 🌍 To run the project 

Question 1: 
```python
python lm_classifier.py --model RNN
```

Question 2:
```py
python lm.py --model RNN
```

## 🧠 Answered Solutions

The project solutions are presented in the models.py file
 - Architecture - GRU 
 - Library - PyTorch