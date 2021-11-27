from flask import Flask, jsonify, request, render_template
import torch.nn as nn
import torch.nn.functional as F
import torch, pickle
import numpy as np

with open('variables.pkl', 'rb') as file:
    var = pickle.load(file)

class BiLSTMTagger(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
    super(BiLSTMTagger, self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
    self.hidden2tag = nn.Linear(hidden_dim*2, target_size)
      
  def forward(self, sentence):
    embeds = self.word_embeddings(sentence)
    lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
    tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    tag_scores = F.log_softmax(tag_space, dim=1)
    return tag_scores

loaded_model = BiLSTMTagger(64, 64, len(var['chars']), len(var['tags']))
loaded_model.load_state_dict(torch.load('joint-bi-lstm-model'))
loaded_model.eval()

'''
def predict(model, senc, tags, chars, char_to_index_dict):
  tag_score = model(torch.from_numpy(np.array([char_to_index_dict[c] for c in list(senc) if c in chars])))
  _, idxes = torch.max(tag_score, 1)
  predicted_tags = [tags[i.item()] for i in idxes]
  token = ''
  prev_tag = 'NC'
  result_tokens = []
  result_tags = []
  for i, tag in enumerate(predicted_tags):
    if tag != 'NC':
      if token:
        result_tokens.append(token)
        result_tags.append(prev_tag)
      token = senc[i]
      prev_tag = tag
    else:
      token += senc[i]
  if token:
    result_tokens.append(token)
    result_tags.append(prev_tag)
  return result_tokens, result_tags
'''

def predict(model, senc, tags, chars, char_to_index_dict):
  tag_score = model(torch.from_numpy(np.array([char_to_index_dict[c] for c in list(senc) if c in chars])))
  _, idxes = torch.max(tag_score, 1)
  predicted_tags = [tags[i.item()] for i in idxes]
  tokens = []
  token = ''
  prev_tag = 'NC'
  for i, tag in enumerate(predicted_tags):
    if tag != 'NC':
      if token:
        tokens.append([token, prev_tag])
      token = senc[i]
      prev_tag = tag
    else:
      token += senc[i]
  if token:
    tokens.append([token, prev_tag])
  return tokens

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
  if request.method == 'POST':
    sentence = request.form['sentence']
    tokens = predict(loaded_model, sentence, var['tags'], var['chars'], var['char_to_index'])
    return render_template('result.html', tokens=tokens)
  return render_template('form.html')

@app.route('/tag', methods=['POST'])
def tag_api():
  sentence = request.form['sentence']
  tokens, tags = predict(loaded_model, sentence, var['tags'], var['chars'], var['char_to_index'])
  return jsonify(list(zip(tokens, tags)))