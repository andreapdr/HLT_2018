import sys
import re
import emoji
import multiprocessing
import unicodedata
import os
import warnings
import pandas as pd
from pytorch_transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle
from datetime import datetime
import spacy


# Rimuovi emoji
def my_demojize(string):
    def replace(match):
        return UNICODE_EMOJI_MY.get(match.group(0), match.group(0))

    return re.sub("\ufe0f", "", EMOJI_REGEXP.sub(replace, string))

# Normalizza testo, stopwords, emoji, parole censurate
def _normalize(text):
    text = my_demojize(text)

    text = RE_SPACE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.translate(TABLE)
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    for pattern, repl in REGEX_REPLACER:
        text = pattern.sub(repl, text)

    return text


def normalize(text, n_workers=2):
    with multiprocessing.Pool(processes=n_workers) as pool:
      text_list = pool.map(_normalize, text.comment_text.tolist())
    return text_list


def preprocess_data(path, filename, val_set_split, load_size=100, max_len=128, batch_size=16, save_it=False):
    dirs = os.listdir(path)
    if 'data_dump.pickle' in dirs and save_it==False:
        with open('data/data_dump.pickle', 'rb') as f:
            data_load = pickle.load(f)
            train_dataloader = data_load['train_set']
            validation_dataloader = data_load['val_set']
            return train_dataloader, validation_dataloader

    from keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split

    df_train = pd.read_csv(path + filename)

    data_test = df_train[:load_size]
    print(f'Dataset length is: {len(data_test)}')
        
    sentences = normalize(data_test, n_workers=4)

    labels = data_test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    X = [tokenizer.tokenize(x) for x in sentences]

    for i, x in enumerate(X):
        if len(x)>512:
            X[i] = x[:511]
            X[i].append('[CLS]')

    # Pad sequences to MAX_LEN via keras preprocessing module
    MAX_LEN = max_len
    X = pad_sequences([tokenizer.convert_tokens_to_ids(x) for x in X], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Build attention masks
    attention_masks = []

    for x in X:
        seq_mask = [float(i>0) for i in x]
        attention_masks.append(seq_mask)

    # Build training and validation set from X
    X_train, X_val, Y_train, Y_val = train_test_split(X, labels, random_state=1608, test_size=val_set_split)
    X_train_masks, X_val_masks, _, _ = train_test_split(attention_masks, X, random_state=1608, test_size=val_set_split)

    print(f'\nTraining set shape: {X_train.shape}\nValidation set shape: {X_val.shape}\n')

    # Convert numpy array into tensor
    X_train_torch = torch.tensor(X_train)
    X_val_torch = torch.tensor(X_val)
    Y_train_torch = torch.tensor(Y_train)
    Y_val_torch = torch.tensor(Y_val)
    X_train_masks_torch = torch.tensor(X_train_masks)
    X_val_masks_torch = torch.tensor(X_val_masks)

    # batch size for fine-tuning BERT on a specific
    # task. It is recommend by the authors a batch size of 16 or 32
    batch_size = batch_size

    # Create an iterator with torch DataLoader.
    train_data = TensorDataset(X_train_torch, X_train_masks_torch, Y_train_torch)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(X_val_torch, X_val_masks_torch, Y_val_torch)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    if save_it:
        with open(path + 'training_local_' + str(max_len) + '_' + str(batch_size) + '.pickle' , 'wb') as f:
            to_dump = {'train_set':train_dataloader, 'val_set':validation_dataloader}
            pickle.dump(to_dump, f)


def preprocess_TEST(path, filename, val_set_split, load_size=100, max_len=128, batch_size=16, save_it=False):
    import numpy as np
    df_train = pd.read_csv(path + filename)

    from keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split

    # data_test = df_train[:load_size]
    data_test = df_train
    print(data_test.head())
    print(len(data_test))
    
    IDs = data_test['id'].values
    sentences = normalize(data_test, n_workers=4)
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    X = [tokenizer.tokenize(x) for x in sentences]

    for i, x in enumerate(X):
        if len(x)>512:
            X[i] = x[:511]
            X[i].append('[CLS]')

    # Pad sequences to MAX_LEN via keras preprocessing module
    MAX_LEN = max_len
    X = pad_sequences([tokenizer.convert_tokens_to_ids(x) for x in X], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Build attention masks
    attention_masks = []

    for x in X:
        seq_mask = [float(i>0) for i in x]
        attention_masks.append(seq_mask)

    X_test = np.array(X)
    X_test_masks = np.array(attention_masks)
    
    print(f'types: {type(X_test), type(X_test_masks), type(IDs)}')
    
    print(f'\nTest shape: {X_test.shape}\nTest Masks set shape: {X_test_masks.shape}\n')
    
    with open('/content/gdrive/My Drive/hlt_project_data/toxic_comment_clf/_tmp_testing.pickle', 'wb') as f:
          pickle.dump([X_test, X_test_masks, IDs], f)
        
    # Convert numpy array into tensor
    X_test_torch = torch.tensor(X_test)
    X_test_masks = torch.tensor(X_test_masks)
    IDs_torch = torch.tensor(IDs)

    # batch size for fine-tuning BERT on a specific
    # task. It is recommend by the authors a batch size of 16 or 32
    batch_size = batch_size

    # Create an iterator with torch DataLoader.
    test_data = TensorDataset(X_test_torch, X_test_masks, IDs_torch)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    if save_it:
        with open(path + 'test_nostop_29_10' + str(max_len) + '_' + str(batch_size) + '.pickle' , 'wb') as f:
            pickle.dump(test_dataloader, f)

def run(load_size, max_len, batch_size):
    filename = opt.filename
    load_size = opt.load_size 
    max_len = opt.max_len 
    batch_size = opt.batch_size
    val_set_split = opt.val_set_split
    save_it = True
    preprocess_data(PATH, filename, val_set_split, load_size, max_len, batch_size,save_it)

parser = argparse.ArgumentParser(description='HLT Project 2019 - preprocessor')

parser.add_argument('--filename', type=str, default='train.csv', help='dataset name')
parser.add_argument('--load_size', type=int, default=10000, help='number of records to retain from whole dataset')
parser.add_argument('--max_len', type=int, default=512, help='max number of token per sentencen (max 512)')
parser.add_argument('--batch_size' type=int, default=32, help='batch size')
parser.add_argument('--val_set_split', type=float, default=0.25, help='training - validation set ratio')
parser.add_argument('--remove_stopwords', type=bool, default=False, help='whether to remove or not stopwords (default: False)')

opt = parser.parse_args()

PATH = '/dataset/'


CUSTOM_TABLE = str.maketrans(
    {
        "\xad": None,
        "\x7f": None,
        "\ufeff": None,
        "\u200b": None,
        "\u200e": None,
        "\u202a": None,
        "\u202c": None,
        "‘": "'",
        "’": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "ɢ": "G",
        "ɪ": "I",
        "ɴ": "N",
        "ʀ": "R",
        "ʏ": "Y",
        "ʙ": "B",
        "ʜ": "H",
        "ʟ": "L",
        "ғ": "F",
        "ᴀ": "A",
        "ᴄ": "C",
        "ᴅ": "D",
        "ᴇ": "E",
        "ᴊ": "J",
        "ᴋ": "K",
        "ᴍ": "M",
        "Μ": "M",
        "ᴏ": "O",
        "ᴘ": "P",
        "ᴛ": "T",
        "ᴜ": "U",
        "ᴡ": "W",
        "ᴠ": "V",
        "ĸ": "K",
        "в": "B",
        "м": "M",
        "н": "H",
        "т": "T",
        "ѕ": "S",
        "—": "-",
        "–": "-",
    }
)

WORDS_REPLACER = [
    ("sh*t", "shit"),
    ("s**t", "shit"),
    ("f*ck", "fuck"),
    ("fu*k", "fuck"),
    ("f**k", "fuck"),
    ("f*****g", "fucking"),
    ("f***ing", "fucking"),
    ("f**king", "fucking"),
    ("p*ssy", "pussy"),
    ("p***y", "pussy"),
    ("pu**y", "pussy"),
    ("p*ss", "piss"),
    ("b*tch", "bitch"),
    ("bit*h", "bitch"),
    ("h*ll", "hell"),
    ("h**l", "hell"),
    ("cr*p", "crap"),
    ("d*mn", "damn"),
    ("stu*pid", "stupid"),
    ("st*pid", "stupid"),
    ("n*gger", "nigger"),
    ("n***ga", "nigger"),
    ("f*ggot", "faggot"),
    ("scr*w", "screw"),
    ("pr*ck", "prick"),
    ("g*d", "god"),
    ("s*x", "sex"),
    ("a*s", "ass"),
    ("a**hole", "asshole"),
    ("a***ole", "asshole"),
    ("a**", "ass"),
]

REGEX_REPLACER = [
    (re.compile(pat.replace("*", "\*"), flags=re.IGNORECASE), repl)
    for pat, repl in WORDS_REPLACER
]

RE_SPACE = re.compile(r"\s")
RE_MULTI_SPACE = re.compile(r"\s+")

NMS_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode + 1) if unicodedata.category(chr(i)) == "Mn"
)

HEBREW_TABLE = {i: "א" for i in range(0x0590, 0x05FF)}
ARABIC_TABLE = {i: "ا" for i in range(0x0600, 0x06FF)}
CHINESE_TABLE = {i: "是" for i in range(0x4E00, 0x9FFF)}
KANJI_TABLE = {i: "ッ" for i in range(0x2E80, 0x2FD5)}
HIRAGANA_TABLE = {i: "ッ" for i in range(0x3041, 0x3096)}
KATAKANA_TABLE = {i: "ッ" for i in range(0x30A0, 0x30FF)}

TABLE = dict()
TABLE.update(CUSTOM_TABLE)
TABLE.update(NMS_TABLE)
# Non-english languages
TABLE.update(CHINESE_TABLE)
TABLE.update(HEBREW_TABLE)
TABLE.update(ARABIC_TABLE)
TABLE.update(HIRAGANA_TABLE)
TABLE.update(KATAKANA_TABLE)
TABLE.update(KANJI_TABLE)


EMOJI_REGEXP = emoji.get_emoji_regexp()

UNICODE_EMOJI_MY = {
    k: f" EMJ {v.strip(':').replace('_', ' ')} "
    for k, v in emoji.UNICODE_EMOJI_ALIAS.items()
}



run()


'''

print(torch.__version__)

with open('/content/gdrive/My Drive/hlt_project_data/toxic_comment_clf/_tmp_testing.pickle', 'rb') as f:
    indata = pickle.load(f)

X_test = indata[0]
X_test_masks = indata[1]
IDs = indata[2]

# lookup table IDS-index 
idx2IDs = {k:v for k,v in enumerate(IDs)}
IDs2idx = {v:k for k,v in enumerate(IDs)}

IDs_converted = [IDs2idx[ID] for ID in IDs]

X_test_torch = torch.tensor(X_test)
X_test_masks = torch.tensor(X_test_masks)
IDs_torch = torch.tensor(IDs_converted)

 # batch size for fine-tuning BERT on a specific
# task. It is recommend by the authors a batch size of 16 or 32
batch_size = 8

# Create an iterator with torch DataLoader.
test_data = TensorDataset(X_test_torch, X_test_masks, IDs_torch)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

with open(PATH + 'test_dataloader_nostop_' + str(batch_size) + '.pickle' , 'wb') as f:
    pickle.dump([test_dataloader, idx2IDs], f)

X_test[0]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.convert_ids_to_tokens(X_test[0])

"""## Testing"""

# test = pd.read_csv(PATH+'df_train_nostop.csv')
flag = False
for i in range(len(test2)):
    if type(test2.iloc[i]['comment_text'])!=str:
        flag=True
        print(i)
        
flag

test2 = test[pd.notnull(test['comment_text'])]

test2.to_csv(PATH + '/df_train_nostop.csv')

df_train.iloc[125028]

PATH

import pandas as pd
PATH = '/content/gdrive/My Drive/hlt_project_data/toxic_comment_clf/'

df_test = pd.read_csv(PATH + 'test.csv')

import spacy
nlp = spacy.load('en_core_web_sm')

to_convert = df_test['comment_text']
len(to_convert)

converted = []
_tot = len(to_convert)
for i, sent in enumerate(to_convert):
    print(f'{i+1}/{_tot}')
    doc = nlp(sent)
    doc = [token.text for token in doc if not token.is_stop and not token.is_punct]
    doc = " ".join(doc)
    converted.append(doc)

import pickle

with open(PATH + 'dump_nostop_test.pickle', 'wb') as f:
    pickle.dump(converted, f)

PATH

with open('/content/gdrive/My Drive/hlt_project_data/toxic_comment_clf/dump_nostop_test.pickle', 'rb') as f:
    converted = pickle.load(f)

converted[0]

to_convert[0]

df_test.head()

df_nostop_test = df_test.copy()
df_nostop_test['comment_text'] = converted
df_nostop_test.head()

# test = pd.read_csv(PATH+'df_train_nostop.csv')
flag = False
for i in range(len(df_nostop_test)):
    if len(df_nostop_test.iloc[i]['comment_text'].split()) == 0:
        df_nostop_test.iloc[i]['comment_text'] = 'love'
        flag=True
        print(i)
        
flag

df_nostop_test.iloc[152493]

df_test.iloc[152493]

df_nostop_test.to_csv(PATH + 'test_nostop.csv', index=False)

with open(PATH + '/dump_nostop.pickle', 'rb') as f:
    converted_df = pickle.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X = [tokenizer.tokenize(x) for x in converted_df]

lenX = [len(x) for x in X]

mean = sum(lenX) / len(lenX)
mode = max(set(lenX), key=lenX.count)
print(f'Mode: {mode}, Mean: {round(mean)}')

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.hist(lenX, range=(0, 250), bins=250)
plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(mode, color='r', linestyle='dashed', linewidth=1, label='Mode')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.title('Sentence Length Histogram no stopwords')
plt.legend()
plt.show()

for i in range(len(X)):
    if len(X[i]) > 128:
        print(i, X[i])
        print(len(X[i]))
        print(df_train.iloc[i][['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

df_train['comment_text'] = converted_df

df_train.to_csv(PATH + '/df_train_nostop.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_stop = [tokenizer.tokenize(x) for x in df_train['comment_text']]

lenX2 = [len(x) for x in X_stop]
mean = sum(lenX2) / len(lenX2)
mode = max(set(lenX2), key=lenX2.count)
print(f'Mode: {mode}, Mean: {round(mean)}')

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.hist(lenX, range=(0, 250), bins=250, alpha=0.5, label='No stopwords')
plt.hist(lenX2, range=(0, 250), bins=250, alpha=0.5, label='Yes stopwords')
# plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='Mean')
# plt.axvline(mode, color='r', linestyle='dashed', linewidth=1, label='Mode')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.title('Sentence Length Histogram')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.hist(lenX2, range=(0, 250), bins=250)
plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(mode, color='r', linestyle='dashed', linewidth=1, label='Mode')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.title('Sentence Length Histogram yes stopwords')
plt.legend()
plt.show()

type(df_test_nostop.iloc[0]['comment_text'])

to_converted

_df = pd.read_csv(PATH + 'test_nostop.csv')

_df.head()

'''