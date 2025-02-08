# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, num_positions)
        
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        
        self.classifier = nn.Linear(d_model, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, indices):
        x = self.embedding(indices)
        x = self.pos_encoding(x)
        
        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attention_maps.append(attn)
            
        logits = self.classifier(x)
        log_probs = self.softmax(logits)
        
        return log_probs, attention_maps


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super(TransformerLayer, self).__init__()
        
        self.d_internal = d_internal
        self.d_model = d_model

        self.query_layer = nn.Linear(d_model, d_internal)
        self.key_layer = nn.Linear(d_model, d_internal)
        self.value_layer = nn.Linear(d_model, d_internal)
        self.softmax = nn.Softmax(dim=-1)
        self.output_projection = nn.Linear(d_internal, d_model)
        
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
    
    def forward(self, input_vecs):
        queries = self.query_layer(input_vecs)
        keys = self.key_layer(input_vecs)
        values = self.value_layer(input_vecs)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.d_internal)
        attention_weights = self.softmax(scores)
        context = torch.matmul(attention_weights, values)

        context = self.output_projection(context)
        output = context + input_vecs
        output = self.fc2(torch.relu(self.fc1(output))) + output
        return output, attention_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        input_size = x.shape[-2]
        indices_to_embed = torch.arange(0, input_size, device=x.device)
        emb = self.emb(indices_to_embed)
        if self.batched:
            emb = emb.unsqueeze(0)
            return x + emb
        else:
            return x + emb


def train_classifier(args, train, dev, batch_size=64):
    vocab_size = 27  # a to z plus space
    num_positions = 20
    d_model = 128
    d_internal = 64
    num_classes = 3
    num_layers = 4
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0.0

        # Shuffle training data
        random.shuffle(train)
        batch_losses = []

        # Process in batches
        for i in range(0, len(train), batch_size):
            batch = train[i:i + batch_size]
            batch_input_tensors = [ex.input_tensor for ex in batch]
            batch_output_tensors = [ex.output_tensor for ex in batch]

            # Pad sequences to max length in batch if needed and stack
            padded_inputs = nn.utils.rnn.pad_sequence(batch_input_tensors, batch_first=True)
            padded_outputs = nn.utils.rnn.pad_sequence(batch_output_tensors, batch_first=True)

            optimizer.zero_grad()
            log_probs, _ = model(padded_inputs)
            loss = loss_fcn(log_probs.view(-1, num_classes), padded_outputs.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())

        avg_loss = total_loss / len(train)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

    model.eval()
    return model


def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
