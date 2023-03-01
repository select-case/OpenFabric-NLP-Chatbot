import pandas as pd
import numpy as np
import requests
import model.intent as intent
from bs4 import BeautifulSoup
import torch.nn as nn
from torch import tensor as tensor
from torch import no_grad
import requests
from bs4 import BeautifulSoup
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset



class DatasetClass(Dataset):
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, header=0)
        
    def __getitem__(self, idx):
        return self.data.y[idx], self.data.x[idx]
    
    def __len__(self):
        return len(self.data)

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    

dataset = DatasetClass("model/intents.csv")
tokenizer = get_tokenizer('basic_english')
train_iter = iter(dataset)



def yield_tokens(data_iter):
    for idx in range(data_iter.__len__()):
        yield tokenizer(dataset[idx][1])
        # yield tokenizer(data_iter[idx][1])

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
train_iter = dataset
num_class = 3
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

import time

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count
    
EPOCHS = 100 # epoch
LR = 2  # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter = dataset
test_iter = dataset
train_dataset = dataset
test_dataset = dataset[26:]
# train_dataset = to_map_style_dataset(train_iter)
# test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    if epoch == 100:
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)






label_list = {
    0: "hello",
    1: "question",
    2: "bye",
}


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()



standard_replies = {
    "hello":"Hi There Ask me a Question",
    "bye":"That was nice. TTYL"
}


def question(str):
    headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}
    google_search_url = "https://www.google.com/search?q="+str
    response =  requests.get(google_search_url, headers = headers)
    html_page = BeautifulSoup(response.text,"html.parser")
    h3_tags = html_page.find_all("h3")
    h3_tags_text = [i.getText() for i in h3_tags]
    wikipedia_answer = 1 if "Description" in h3_tags_text else 0
    featured = 1 if "About featured snippets" in html_page.body.text else 0
    

    
    if wikipedia_answer:
        index = h3_tags_text.index('Description')
        answer = h3_tags[index].find_parent().find('span').text
    
    elif featured:
        answer = html_page.find(text = "About featured snippets").find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_all('span')[0].text
        if('\n' in answer):
            answer.replace('\n',' ')

    else:
        answer_url = h3_tags[0].find_parent().find_parent().find('a',href=True)['href']
        response_ans =  requests.get(answer_url, headers = headers)
        answer = "I'm not sure I can properly answer that, you can try visiting "
        answer += answer_url

    print(type(answer))
    return answer


def start(s):
    # model = torch.load('model/state_dict_model.pth', map_location='cpu')
    category = label_list[predict(s, text_pipeline)]
    if category == "question":
        answer = question(s)
    else:
        answer = standard_replies[category]
    return answer