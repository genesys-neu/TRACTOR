import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math
from vit_pytorch import ViT

# Define model
class ConvNN(nn.Module):
    def __init__(self, numChannels=1, slice_len=4, num_feats=17, classes=4):
        super(ConvNN, self).__init__()

        self.numChannels = numChannels

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
                               kernel_size=(4, 1))
        self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2))
        ##  initialize second set of CONV => RELU => POOL layers
        # self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
        #                    kernel_size=(5, 5))
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        ## initialize first (and only) set of FC => RELU layers

        # pass a random input
        rand_x = torch.Tensor(np.random.random((1, 1, slice_len, num_feats)))
        output_size = torch.flatten(self.conv1(rand_x)).shape
        self.fc1 = nn.Linear(in_features=output_size.numel(), out_features=512)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=512, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.reshape \
            ((x.shape[0], self.numChannels, x.shape[1], x.shape[2]))   # CNN 2D expects a [N, Cin, H, W] size of data
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.maxpool1(x)
        ## pass the output from the previous layer through the second
        ## set of CONV => RELU => POOL layers
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        ## flatten the output from the previous layer and pass it
        ## through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


class TransformerNN(nn.Module):
    def __init__(self, classes: int = 4, num_feats: int = 17, slice_len: int = 32, nhead: int = 1, nlayers: int = 2,
                 dropout: float = 0.2, use_pos: bool = False, custom_enc: bool = False):
        super(TransformerNN, self).__init__()

        if use_pos and not custom_enc:
            num_feats = num_feats - 1 # exclude the timestamp column (0) that will be used for positional encoding

        self.norm = nn.LayerNorm(num_feats)
        # create the positional encoder
        self.use_positional_enc = use_pos

        # TODO not entirely sure why we need d_model = num_feats + 1 for tradtional pos. encoder
        self.pos_encoder = PositionalEncoding(num_feats + 1, dropout, custom_enc=custom_enc) if use_pos else None
        # define the encoder layers

        encoder_layers = TransformerEncoderLayer(d_model=num_feats, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = num_feats

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(num_feats*slice_len, 256)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(256, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classes log probabilities
        """
        # src = self.norm(src) should not be necessary since output can be already normalized
        # apply positional encoding if decided
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        # pass through encoder layers
        t_out = self.transformer_encoder(src)
        # flatten already contextualized KPIs
        t_out = torch.flatten(t_out, start_dim=1)
        # Pass through MLP classifier
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.logSoftmax(output)
        return output


class TransformerNN_v2(nn.Module):
    def __init__(self, classes: int = 4, num_feats: int = 17, slice_len: int = 32, nhead: int = 1, nlayers: int = 2,
                 dropout: float = 0.2, use_pos: bool = False):
        super(TransformerNN_v2, self).__init__()

        self.norm = nn.LayerNorm(num_feats)
        # create the positional encoder
        self.use_positional_enc = use_pos
        self.pos_encoder = PositionalEncoding(num_feats + 1, dropout)
        # define [CLS] token to be used for classification
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, num_feats))
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(num_feats, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = num_feats

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(num_feats, num_feats *2)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(num_feats *2, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classes log probabilities
        """
        cls_tokens = self.cls_token.repeat(src.size(0) ,1 ,1)
        src = torch.column_stack((cls_tokens, src))
        # src = self.norm(src) should not be necessary since output can be already normalized
        # apply positional encoding if decided
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        t_out = self.transformer_encoder(src)
        # get hidden state of the [CLS] token
        t_out = t_out[:, 0, :].squeeze()
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.logSoftmax(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000, custom_enc=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.custom_enc = custom_enc

        if not self.custom_enc:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            # ToDo: try the following change
            # pe = torch.zeros(max_len, 1, d_model)
            # pe[:, 0, 0::2] = torch.sin(position * div_term)
            # pe[:, 0, 1::2] = torch.cos(position * div_term)
            # try the following instead
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe[:, :-1]
            pe = pe.unsqueeze(0)

            self.register_buffer('pe', pe)
            """
            import matplotlib.pyplot as plt
            np_pe = np.array(pe[0])
            plt.imshow(np_pe, aspect='auto')
            plt.colorbar()
            plt.show()
            """
        else:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model-1))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, features]
        """

        if not self.custom_enc:
            x = x + self.pe[:, :x.size(1)]

        else:

            rel_time_ix_info = x[:, :, 0]
            x = x[:, :, 1:]
            """
            # alternative method to compute this, but below should be faster
            all_pe = torch.zeros((x.shape[0], x.shape[1], x.shape[2]))
            for s in range(x.shape[0]): # iterate over sample in batch
                s_timeinfo_ix = torch.clip(rel_time_ix_info[s], max=self.max_len-1).to(torch.long)
                all_pe[s] = self.pe[:, s_timeinfo_ix]
            """
            all_pe = torch.stack(
                [self.pe[:, torch.clip(s, max=self.max_len-1).to(torch.long)]
                    for s in torch.unbind(rel_time_ix_info, dim=0)]
                , dim=0).squeeze()

            if x.device.type == 'cuda':
                all_pe = all_pe.to(x.device)
            x = x + all_pe

        return self.dropout(x)


class TransformerNN_old(nn.Module):
    def __init__(self, classes: int = 4, num_feats: int = 17, slice_len: int = 32, nhead: int = 1, nlayers: int = 2,
                 dropout: float = 0.2, use_pos: bool = False):
        super(TransformerNN_old, self).__init__()
        self.norm = nn.LayerNorm(num_feats)
        # create the positional encoder
        self.use_positional_enc = use_pos
        self.pos_encoder = PositionalEncoding(num_feats + 1, dropout, max_len=5000)
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(num_feats, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = num_feats

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(num_feats *slice_len, 256)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(256, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classes log probabilities
        """
        # src = self.norm(src) should not be necessary since output can be already normalized
        # apply positional encoding if decided
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        # pass through encoder layers
        t_out = self.transformer_encoder(src)
        # flatten already contextualized KPIs
        t_out = torch.flatten(t_out, start_dim=1)
        # Pass through MLP classifier
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.logSoftmax(output)
        return output


class megatron_ViT(ViT):
    def __init__(self, classes: int = 4,  num_feats: int = 17, slice_len: int = 32, dropout=0.25):
        patch_Tsize = 4
        super(megatron_ViT, self).__init__(
            image_size=(slice_len, num_feats),
            patch_size=(patch_Tsize, num_feats),
            num_classes=classes,
            channels=1,
            dim=256,
            heads=8,
            depth=2,
            mlp_dim=2048,
            dropout=dropout,
            emb_dropout=dropout
        )

