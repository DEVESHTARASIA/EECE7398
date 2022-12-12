from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
from io import open
import random
import os
from collections import Counter
import argparse
import time
import math
import unicodedata
import string
import re


#Calculates BLeu Score using method1 smoothing
def calculate_bleu_score(encoder, decoder, pairs):
    smooth = SmoothingFunction().method1
    bleu_score = []
    for pair in pairs:
        try:
            output_words, attentions = evaluate(encoder, decoder, pair[0])
        except RuntimeError:
            pass
        output_sentence = ' '.join(output_words)
        print(output_sentence)

        bleu_score.append(sentence_bleu([output_sentence], pair[1], smoothing_function=smooth))
    bleu_score_mean = np.mean(bleu_score)
    return bleu_score_mean

#This is based on PyTorch's tutorial on seq2seq NMT with an added attention layer

class Lang:
    def __init__(self, name, vocab_path):
        self.name = name
        self.word2index = {'<unk>':2}
        self.index2word = {0: "sos", 1: "eos", 2: '<unk>'}
        self.n_words = 3 # Count <unk>, sos and eos
        self.vocab_path = vocab_path
        
    def loadVocab(self):
        with open(self.vocab_path) as f:
            rawData = f.readlines()
            vocab = list(map(lambda word: word[:-1], rawData))
        vocab_norm = [normalize_String(word) for word in vocab]
        unique_vocab_norm = list(Counter(vocab_norm).keys())
        for word in unique_vocab_norm:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def make_indices_from_sentences(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        else:
            indexes.append(lang.word2index['<unk>'])
    return indexes

def make_tensors_from_sentences(lang, sentence):
    indexes = make_indices_from_sentences(lang, sentence)
    indexes.append(eos_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def make_tensors_from_pair(pair):
    input_tensor = make_tensors_from_sentences(input_lang, pair[0])
    target_tensor = make_tensors_from_sentences(output_lang, pair[1])
    return (input_tensor, target_tensor)


def convert_Unicode_to_Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_String(s):
    s = convert_Unicode_to_Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_both_languages(lang1, lang2, vocab1=None, vocab2=None, reverse=False):
  
    lines1 = open(lang1, encoding='utf-8').read().strip().split('\n')
    lines2 = open(lang2, encoding='utf-8').read().strip().split('\n')

    lines = [line1 + '\t' + line2 for line1, line2 in zip(lines1, lines2)]

    pairs = [[normalize_String(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2, vocab2)
        output_lang = Lang(lang1, vocab1)
    else:
        input_lang = Lang(lang1, vocab1)
        output_lang = Lang(lang2, vocab2)

    return input_lang, output_lang, pairs

def filterPair(p, MAX_LENGTH=50):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs, max_length=50):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepare_data_for_model(lang1, lang2, vocab1=None, vocab2=None, reverse=False, max_length=50):
    input_lang, output_lang, pairs = read_both_languages(lang1, lang2, vocab1=vocab1, vocab2=vocab2, reverse=reverse)
    pairs = filterPairs(pairs, max_length)
    input_lang.loadVocab()
    output_lang.loadVocab()
    return input_lang, output_lang, pairs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionUnit(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=50):
        super(AttentionUnit, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[sos_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else: #NO Teacher Forcing
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == eos_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def start_training(encoder, decoder, n_iters, learning_rate=0.01):
    old_loss = 99999
    if not os.path.isdir('model'):
        os.mkdir('model')
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [make_tensors_from_pair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    
    for iter in range(1, n_iters+1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if(iter%100==0):
            print('Iteration {} Loss = {}'.format(iter/100,loss))

        if iter >1:
            if old_loss > loss: #save the better model
                print('Saving model with loss', loss)
                torch.save(encoder.state_dict(), 'model/encoderVi.ckpt')
                torch.save(decoder.state_dict(), 'model/decoderVi.ckpt')
                old_loss=loss
                
def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = make_tensors_from_sentences(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        max_length = decoder.max_length
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[sos_token]], device=device)  # sos

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == eos_token:
                decoded_words.append('<eos>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NMT')
    parser.add_argument('mode', type=str, help='train for training \n test for testing \n translate for translating')

    arg = parser.parse_args()

    device = torch.device("cuda:"+arg.gpu if torch.cuda.is_available() else "cpu")
    sos_token = 0
    eos_token = 1

    MAX_LENGTH = 50
    teacher_forcing_ratio = 0.3
    model_name = 'Vietnamese'
    print(os.getcwd())
    TRAIN_PATH = ['data/train_en.txt', 'data/train_vi.txt']
    VOCAB_PATH = ['data/vocab_en.txt', 'data/vocab_vi.txt']
    TEST_PATH = ['data/tst2012_en.txt', 'data/tst2012_vi.txt']
    
    input_lang, output_lang, pairs = prepare_data_for_model(TRAIN_PATH[0],TRAIN_PATH[1], VOCAB_PATH[0], VOCAB_PATH[1], False, max_length=MAX_LENGTH)

    hidden_size = 256
    my_encoder = Encoder(input_lang.n_words, hidden_size).to(device)
    my_decoder = AttentionUnit(hidden_size, output_lang.n_words, dropout_p=0.5).to(device)

    if arg.mode=='train':
        start_training(my_encoder, my_decoder, 10000)
        print("Model saved")


    elif arg.mode=='test':
        my_encoder.load_state_dict(torch.load('model/encoderVi.ckpt', map_location=torch.device('cpu')))
        my_decoder.load_state_dict(torch.load('model/decoderVi.ckpt', map_location=torch.device('cpu')))
        _, _, test_pair = prepare_data_for_model(TEST_PATH[0],TEST_PATH[1], VOCAB_PATH[0], VOCAB_PATH[1], False, max_length=MAX_LENGTH)
        print("BLEU score:"+str(calculate_bleu_score(my_encoder, my_decoder, test_pair)))
    
    elif arg.mode=='translate':
        print("Enter the english sentence to be translated")
        string = str(input())
        my_encoder.load_state_dict(torch.load('model/encoderVi.ckpt', map_location=torch.device('cpu')))
        my_decoder.load_state_dict(torch.load('model/decoderVi.ckpt', map_location=torch.device('cpu')))
        decoded_words, _ = evaluate(my_encoder, my_decoder, string)
        output_sentence = ' '.join(decoded_words)
        print(output_sentence)