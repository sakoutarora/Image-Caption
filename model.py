import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.incep = models.inception_v3(pretrained=True, aux_logits=False)
        for layer in self.incep.parameters():
            layer.requires_grad = False
        self.fc = nn.Linear(2048, embed_size)
        self.relu = nn.ReLU(inplace=True)
        self.incep.fc = self.fc
        self.dropout = nn.Dropout(0.3)

    def forward(self, img):
        return self.dropout(self.relu(self.incep(img)))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.vocal_size = vocab_size
        self.hids = hidden_size
        self.LSTM = nn.LSTM(embed_size, self.hids, num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.drop = nn.Dropout(0.3)

    def forward(self, features, caption):
        embeddings = self.drop(self.embed(caption.long()))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hidden, c0 = self.LSTM(embeddings)
        output = self.linear(hidden)
        return output


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, vocal_size, hidden_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCnn = EncoderCNN(embed_size)
        self.decoderRnn = DecoderRNN(embed_size, hidden_size, num_layers, vocal_size)

    def forward(self, image, caption):
        features = self.encoderCnn(image)
        out = self.decoderRnn(features, caption)
        return out

    def caption_image(self, image, vocab, max_len = 50):
        result_caption = []
        with torch.no_grad():
            x = self.encoderCnn(image).unsqueeze(0)
            state = None
            for _ in range(max_len):
                hidden, state = self.decoderRnn.LSTM(x, state)
                output = self.decoderRnn.linear(hidden.squeeze(0))
                predict = output.argmax(1)
                result_caption.append(predict.item())
                x = self.decoderRnn.embed(predict).unsquueze(0)
                if vocab.itos[predict.item()] == "<EOS>":
                    break
            return [vocab.itos[idx] for idx in result_caption]

