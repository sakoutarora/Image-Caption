import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from torch.nn.utils.rnn import pad_sequence

base_path = 'F:\\ML\\image-caption-dataset\\flickr30k_images'
comments_path = os.path.join(base_path, 'results.csv')
img_root = os.path.join(base_path, 'flickr30k_images')


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenizer_end(text):
        rg = RegexpTokenizer('[a-zA-Z]+')
        return [word.lower() for word in rg.tokenize(text)]

    def create_vocab(self, dictionary):
        frequencies = {}
        idx = 4
        for sentence in dictionary:
            token = self.tokenizer_end(sentence)
            for word in token:
                word = word.lower()
                if word not in frequencies:
                    frequencies[word] = 0
                elif frequencies[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx = idx + 1
                    frequencies[word] += 1
                else:
                    frequencies[word] += 1

    def convert(self, line):
        token = self.tokenizer_end(line)
        return [
            self.stoi[word.lower()] if word.lower() in self.stoi else self.stoi['<UNK>']
            for word in token
        ]


class FlickDataset(Dataset):
    def __init__(self, root=img_root, caption_path=comments_path, transform=transforms.ToTensor(), freq=5):
        self.caption = pd.read_csv(caption_path, delimiter='|')
        self.caption.fillna(value='NA', inplace=True)
        self.root = root
        self.freq = freq
        self.transform = transform
        self.vocab = Vocabulary(self.freq)
        self.vocab.create_vocab(self.caption[' comment'].tolist())
        self.vocab_len = len(self.vocab)

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.caption.iloc[item, 0])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        comment = self.caption.iloc[item, 2]
        numericalize = [self.vocab.stoi['<SOS>']]
        numericalize += (self.vocab.convert(comment))
        numericalize.append(self.vocab.stoi['<EOS>'])
        return image, torch.Tensor(numericalize)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        target = [item[1] for item in batch]
        target = pad_sequence(target, batch_first=False, padding_value=self.pad_idx)
        return imgs, target


def loader(root=img_root, caption_file=comments_path, batch_size=32, transform=None, shuffle=False,):
    dataset = FlickDataset(root=root, caption_path=caption_file, transform=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=MyCollate(pad_idx=pad_idx))
    return load, dataset

