import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

# Training data
spam_messages = ["Gana dinero rápido", "Reclama tu premio"]
ham_messages = ["Reunión a las 3PM", "Actualización del proyecto necesaria"]

# Combine training data
all_messages = spam_messages + ham_messages
labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)

# Vectorize the messages to get bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(all_messages)

# Extract vocabulary and bigrams
bigrams = vectorizer.get_feature_names_out()

# Create transition matrices for spam and ham
transition_counts_spam = defaultdict(lambda: defaultdict(int))
transition_counts_ham = defaultdict(lambda: defaultdict(int))

# Populate transition counts
for i, message in enumerate(all_messages):
    words = message.split()
    for j in range(len(words) - 1):
        if labels[i] == 'spam':
            transition_counts_spam[words[j]][words[j + 1]] += 1
        else:
            transition_counts_ham[words[j]][words[j + 1]] += 1

# Convert counts to probabilities with Laplace smoothing
vocab_size = len(set(" ".join(all_messages).split()))
transition_probs_spam = defaultdict(lambda: defaultdict(float))
transition_probs_ham = defaultdict(lambda: defaultdict(float))

for word, next_words in transition_counts_spam.items():
    total_count = sum(next_words.values()) + vocab_size  # Add vocab size for smoothing
    for next_word, count in next_words.items():
        transition_probs_spam[word][next_word] = (count + 1) / total_count

for word, next_words in transition_counts_ham.items():
    total_count = sum(next_words.values()) + vocab_size  # Add vocab size for smoothing
    for next_word, count in next_words.items():
        transition_probs_ham[word][next_word] = (count + 1) / total_count

# Step 2: Define the message prediction function
def predict_markov(message, spam_probs, ham_probs, vocab_size, p_spam=0.5, p_ham=0.5):
    words = message.split()
    spam_likelihood = p_spam
    ham_likelihood = p_ham
    
    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        
        # Spam transition
        spam_likelihood *= spam_probs[current_word].get(next_word, 1 / (vocab_size + 1))
        
        # Ham transition
        ham_likelihood *= ham_probs[current_word].get(next_word, 1 / (vocab_size + 1))
    
    return 'spam' if spam_likelihood > ham_likelihood else 'ham', spam_likelihood, ham_likelihood

# Test message
test_message = "Reclama tu dinero premio"
prediction, spam_score, ham_score = predict_markov(test_message, transition_probs_spam, transition_probs_ham, vocab_size)

# Prepare results for display
results_df = pd.DataFrame({
    "Message": [test_message],
    "Spam Score": [spam_score],
    "Ham Score": [ham_score],
    "Prediction": [prediction]
})

display("Markov Model Spam Filter Results", results_df)