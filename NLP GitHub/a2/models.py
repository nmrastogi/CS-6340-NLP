import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import SentimentExample, WordEmbeddings, List

# Edit distance calculation for spelling correction with early stopping
def edit_distance(word1, word2, max_edit_distance=2):
    """
    Compute the Levenshtein distance (edit distance) between two words, with early stopping.
    Stops computation if the distance exceeds max_edit_distance.
    """
    m, n = len(word1), len(word2)
    if abs(m - n) > max_edit_distance:
        return max_edit_distance + 1  # Early stop if length difference is too large

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            
            # Early stopping if distance exceeds the allowed threshold
            if dp[i][j] > max_edit_distance:
                return max_edit_distance + 1

    return dp[m][n]

# Cache for storing already-corrected words
spelling_cache = {}

def correct_spelling(word, indexer, max_edit_distance=2):
    """
    Corrects spelling of a word by finding the closest word in the indexer based on edit distance.
    Uses caching to avoid redundant computation.
    :param word: The misspelled word
    :param indexer: The vocabulary (word indexer)
    :param max_edit_distance: Maximum allowed edit distance for correction
    :return: The corrected word if found, else the original word
    """
    # If word is already corrected in cache, return cached result
    if word in spelling_cache:
        return spelling_cache[word]

    min_distance = float('inf')
    corrected_word = word

    # Limit search to words with similar lengths for optimization
    for candidate_word in indexer.ints_to_objs.values():
        if abs(len(word) - len(candidate_word)) <= max_edit_distance:
            distance = edit_distance(word, candidate_word, max_edit_distance)
            if distance < min_distance and distance <= max_edit_distance:
                min_distance = distance
                corrected_word = candidate_word

    # Cache the result for future lookups
    spelling_cache[word] = corrected_word
    return corrected_word

class SentimentClassifier(object):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        return 1  # Always predicts positive class


class NeuralSentimentClassifier(nn.Module, SentimentClassifier):
    """
    A neural network classifier for sentiment analysis using the Deep Averaging Network (DAN).
    """
    def __init__(self, embedding_vectors, hidden_size, output_size, word_indexer):
        super(NeuralSentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_vectors),
            freeze=False,  # Set to True if you don't want to fine-tune the embeddings
            padding_idx=0  # Assuming PAD token is at index 0
        )
        
        self.fc1 = nn.Linear(self.embedding.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.word_indexer = word_indexer

    def forward(self, word_indices):
        if word_indices.dim() == 1:
            word_indices = word_indices.unsqueeze(0)  # Add batch dimension if missing

        embeddings = self.embedding(word_indices)  # [batch_size, sentence_len, embedding_dim]
        averaged_embeddings = torch.mean(embeddings, dim=1)  # [batch_size, embedding_dim]
        
        out = self.fc1(averaged_embeddings)
        out = self.relu(out)
        out = self.fc2(out)
        return self.softmax(out)

    def predict(self, ex_words, has_typos):
        with torch.no_grad():
            corrected_words = []
            
            if has_typos:
                for word in ex_words:
                    if self.word_indexer.index_of(word) == -1:
                        corrected_word = correct_spelling(word, self.word_indexer)
                        corrected_words.append(corrected_word)
                    else:
                        corrected_words.append(word)
            else:
                corrected_words = ex_words

            word_indices = torch.tensor([
                self.word_indexer.index_of(w) if self.word_indexer.index_of(w) != -1
                else self.word_indexer.index_of("UNK") for w in corrected_words
            ])
            logits = self.forward(word_indices.unsqueeze(0))  # Add batch dimension
            pred = torch.argmax(logits)
            return pred.item()


def precompute_word_indices(examples, word_indexer):
    """
    Precompute word indices for each example to speed up training.
    :param examples: List of SentimentExample
    :param word_indexer: The word indexer for word to index mapping
    :return: List of precomputed word indices and labels
    """
    precomputed_data = []
    for ex in examples:
        word_indices = torch.tensor([
            word_indexer.index_of(w) if word_indexer.index_of(w) != -1
            else word_indexer.index_of("UNK") for w in ex.words
        ])
        precomputed_data.append((word_indices, ex.label))
    return precomputed_data


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    Training function with typo handling for dev-typo.txt data, using a batch size of 32.
    """
    model = NeuralSentimentClassifier(
        embedding_vectors=word_embeddings.vectors,
        hidden_size=args.hidden_size,
        output_size=2,  # Binary classification
        word_indexer=word_embeddings.word_indexer
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    # Precompute word indices to avoid redundant calculations
    precomputed_train_data = precompute_word_indices(train_exs, word_embeddings.word_indexer)

    batch_size = 64  # Fixed batch size of 32
    num_batches = len(precomputed_train_data) // batch_size

    for epoch in range(10):
        model.train()
        total_loss = 0
        random.shuffle(precomputed_train_data)

        for i in range(0, len(precomputed_train_data), batch_size):
            batch = precomputed_train_data[i:i + batch_size]
            word_indices_batch = torch.nn.utils.rnn.pad_sequence([ex[0] for ex in batch], batch_first=True, padding_value=0)
            labels_batch = torch.tensor([ex[1] for ex in batch])

            optimizer.zero_grad()
            output = model(word_indices_batch)
            loss = criterion(output, labels_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{10}, Loss: {total_loss / num_batches}")

    return model
