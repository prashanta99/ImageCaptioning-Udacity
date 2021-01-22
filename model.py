import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size , 
                            num_layers = num_layers,
                            batch_first=True)
        
        self.wordEmbedding = nn.Embedding(vocab_size, embed_size)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, features, captions):
        embeds = self.wordEmbedding(captions[:, :-1])
        features = features.unsqueeze(1)
        stacked_input = torch.cat((features, embeds), dim=1)
        output, hidden = self.lstm(stacked_input)
        output = self.linear(output)
         
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        word_list_ids = []
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states) 
            probs = self.linear(lstm_out)   
            probs, word_idx_dict = probs.max(2)         # Get the highest Probabilitie word
            word_list_ids.append(word_idx_dict.item())
            inputs = self.wordEmbedding(word_idx_dict) 
                
        return word_list_ids
        