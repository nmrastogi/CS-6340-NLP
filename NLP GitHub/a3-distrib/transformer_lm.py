# transformer_lm.py

import torch
import torch.nn as nn
import numpy as np
import time
from transformer import PositionalEncoding

class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, transformer_model, vocab_index):
        super(NeuralLanguageModel, self).__init__()
        self.model = transformer_model
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        if not context:  # Handle empty context by returning uniform distribution
            return np.ones(len(self.vocab_index)) * np.log(1.0 / len(self.vocab_index))

        # Truncate context to the max allowed sequence length (20 characters)
        context = context[-20:]

        # Convert context to tensor
        context_ids = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0)

        # Set the model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            logits = self.model(context_ids)  # Model output: [1, seq_len, vocab_size]
            # Normalize using Softmax to ensure probability distribution
            probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze(0).cpu().numpy()
            log_probs = np.log(probs)

        return log_probs

    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        for next_char in next_chars:
            log_probs = self.get_next_char_log_probs(context)
            char_idx = self.vocab_index.index_of(next_char)
            log_prob += log_probs[char_idx]
            context += next_char  # Update context with next character
        return log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)
    d_model = 128  # Increased dimension
    nhead = 4  # Increased number of heads for better representation learning
    num_layers = 4  # Increased number of layers
    chunk_size = 20
    batch_size = 32
    dropout_rate = 0.1

    # Initialize the Transformer model
    transformer_model = TransformerLanguageModel(vocab_size, d_model, nhead, num_layers, chunk_size, dropout=dropout_rate)
    optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=5e-4)  # Higher initial learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()
        
        for i in range(0, len(train_text) - chunk_size, batch_size * chunk_size):
            batch_chunks = [
                train_text[j:j + chunk_size] 
                for j in range(i, i + batch_size * chunk_size, chunk_size)
                if len(train_text[j:j + chunk_size]) == chunk_size
            ]
            
            if not batch_chunks:  # Skip if no valid chunks in this batch
                continue

            input_ids = torch.tensor([
                [vocab_index.index_of(c) for c in chunk[:-1]] for chunk in batch_chunks
            ], dtype=torch.long)
            target_ids = torch.tensor([
                [vocab_index.index_of(c) for c in chunk[1:]] for chunk in batch_chunks
            ], dtype=torch.long)

            optimizer.zero_grad()
            logits = transformer_model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size

        avg_loss = total_loss / (len(train_text) // chunk_size)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")
        scheduler.step()

    return NeuralLanguageModel(transformer_model, vocab_index)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, chunk_size, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, chunk_size, batched=True)

        # Transformer Encoder with dropout and batch normalization for stability
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.chunk_size)
        x = self.pos_encoding(x)

        # Generate square subsequent mask for causal attention
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        transformer_out = self.transformer(x, mask=mask)
        
        logits = self.fc_out(transformer_out)
        return logits

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask


def evaluate_model(dev_text, neural_model, vocab_index):
    log_prob_sum = 0
    n_tokens = len(dev_text) - 1
    
    for i in range(n_tokens):
        context = dev_text[:i + 1]
        next_char = dev_text[i + 1]
        log_probs = neural_model.get_next_char_log_probs(context)
        next_char_idx = vocab_index.index_of(next_char)
        log_prob_sum += log_probs[next_char_idx]
    
    avg_log_prob = log_prob_sum / n_tokens
    perplexity = np.exp(-avg_log_prob)
    print(f"Average Log Probability: {avg_log_prob:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    return avg_log_prob, perplexity
