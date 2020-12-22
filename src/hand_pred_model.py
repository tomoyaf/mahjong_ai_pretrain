import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertLayer, BertConfig
from glob import glob
import numpy as np
import math
from transformers import AdamW, get_linear_schedule_with_warmup
import catalyst

from hand_pred_data import PaifuDataset


def get_model():
    # tile(37), menzen(2), reach_state(2), n_reach(3),
    # reach_ippatsu(2), dans(21), rates(19), oya(4),
    # scores(13), n_honba(3), n_round(12), sanma_or_yonma(2),
    # han_or_ton(2), aka_ari(2), kui_ari(2), special_token(4)
    vocab_size = 37 + 2 + 2 + 3 + 2 + 21 + 19 + 4 + 13 + 3 + 12 + 2 + 2 + 2 + 2 + 4 + 4 + 6 + 8 # 130 + who(4) + sum_discards(6) + shanten(8)
    max_position_embeddings = 239 # base + who(1) + sum_discards(1) + shanten(1)

    config = {}
    config['vocab_size'] = vocab_size
    config['d_model'] = 512
    config['nhead'] = 8
    config['max_position_embeddings'] = max_position_embeddings
    config['dim_feedforward'] = 2048
    config['num_encoder_layers'] = 6
    config['num_decoder_layers'] = 6
    config['pad_token_id'] = 0
    config['num_outputs'] = 38 # 37 + sep
    config['hidden_dropout_prob'] = 0.1
    config['layer_norm_eps'] = 1e-6

    model = MahjongModel(config)
    return model


def get_optimizer(model, lr=1e-4, weight_decay=0.01, n_epochs=10, n_warmup_steps=1e4, n_training_steps=4e5):
    n_warmup_steps = 4000
    print(f'lr:{lr}, weight_decay:{weight_decay}, n_training_steps:{n_training_steps}, n_warmup_steps:{n_warmup_steps}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.98])
    # optimizer = AdamW(model.parameters(), lr=lr, betas=[0.9, 0.], weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=n_warmup_steps,
    #     num_training_steps=n_training_steps
    # )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min([(step + 1) ** -0.5, (step + 1) * n_warmup_steps ** -1.5])
    )
    return optimizer, lr_scheduler


def process_path_list(path_list, max_data_size, n_max):
    np.random.shuffle(path_list)
    path_list = path_list[:(max_data_size // n_max)]

    data_size = len(path_list) * n_max
    train_size = int(data_size * 0.9)
    val_size = int(data_size * 0.05)
    test_size = data_size - train_size - val_size

    file_train_size = train_size // n_max
    file_val_size = val_size // n_max
    file_test_size = test_size // n_max

    train_path_list = path_list[:file_train_size]
    val_path_list = path_list[file_train_size:file_train_size+file_val_size]
    test_path_list = path_list[-file_test_size:]

    return train_path_list, val_path_list, test_path_list, data_size, train_size, val_size, test_size


def get_loaders(batch_size, model_name, max_data_size, n_max=100):
    train_path_list = []
    val_path_list = []
    test_path_list = []
    data_size = 0
    train_size = 0
    val_size = 0
    test_size = 0
    base_path = './pickle/'
    path_list = glob(f'{base_path}/{model_name}/*')
    train_path_list, val_path_list, test_path_list, data_size, train_size, val_size, test_size = process_path_list(path_list, max_data_size, n_max)

    print(f'Data size : {data_size}, Train size : {train_size}, Val size : {val_size}, Test size : {test_size}')

    train_dataset = PaifuDataset(train_path_list, n_max=n_max)
    val_dataset = PaifuDataset(val_path_list, n_max=n_max)
    test_dataset = PaifuDataset(test_path_list, n_max=n_max)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


class MahjongEmbeddings(nn.Module):
    def __init__(self, config, n_token_type=68):
        super(MahjongEmbeddings, self).__init__()
        print(config)
        self.config = config
        self.symbol_embeddings = nn.Embedding(
            config['vocab_size'],
            config['d_model'],
            padding_idx=config['pad_token_id']
        )
        self.n_position_embeddings = 25 + 20 + 1 # discard : 25, meld : 20, pad
        self.token_type_embeddings = nn.Embedding(
            n_token_type,
            config['d_model'],
            padding_idx=config['pad_token_id']
        )

        decoder_vocab = 37 + 3 # 37 + sep + cls + pad
        self.tgt_embeddings = nn.Embedding(
            decoder_vocab,
            config['d_model'],
            padding_idx=config['pad_token_id']
        )
        self.tgt_pad_token_id = 0
        self.tgt_cls_token_id = 38
        self.tgt_sep_token_id = 39

        self.layer_norm = nn.LayerNorm(config['d_model'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

        # Special token id
        self.sep_token_id = 127
        self.cls_token_id = 128
        self.mask_token_id = 129
        self.pad_token_id = config['pad_token_id']

        # Tile token offset
        self.menzen_offset = 38
        self.reach_state_offset = 40
        self.n_reach_offset = 42
        self.reach_ippatsu_offset = 45
        self.dans_offset = 47
        self.rates_offset = 68
        self.scores_offset = 87
        self.oya_offset = 100
        self.n_honba_offset = 104
        self.n_round_offset = 107
        self.sanma_or_yonma_offset = 119
        self.han_or_ton_offset = 121
        self.aka_ari_offset = 123
        self.kui_ari_offset = 125
        self.shanten_offset = 130
        self.who_offset = 138
        self.sum_discards_offset = 142
        # self.shanten_diff_offset = 146

        # Token type id
        self.hand_token_id = 1
        self.discard_0_token_id = 2
        self.discard_1_token_id = 3
        self.discard_2_token_id = 4
        self.discard_3_token_id = 5
        self.menzen_0_token_id = 6
        self.menzen_1_token_id = 7
        self.menzen_2_token_id = 8
        self.menzen_3_token_id = 9
        self.reach_state_0_token_id = 10
        self.reach_state_1_token_id = 11
        self.reach_state_2_token_id = 12
        self.reach_state_3_token_id = 13
        self.n_reach_token_id = 14
        self.reach_ippatsu_0_token_id = 15
        self.reach_ippatsu_1_token_id = 16
        self.reach_ippatsu_2_token_id = 17
        self.reach_ippatsu_3_token_id = 18
        self.dora_token_id = 19
        self.dans_0_token_id = 20
        self.dans_1_token_id = 21
        self.dans_2_token_id = 22
        self.dans_3_token_id = 23
        self.rates_0_token_id = 24
        self.rates_1_token_id = 25
        self.rates_2_token_id = 26
        self.rates_3_token_id = 27
        self.scores_0_token_id = 28
        self.scores_1_token_id = 29
        self.scores_2_token_id = 30
        self.scores_3_token_id = 31
        self.oya_token_id = 32
        self.n_honba_token_id = 33
        self.n_round_token_id = 34
        self.sanma_or_yonma_token_id = 35
        self.han_or_ton_token_id = 36
        self.aka_ari_token_id = 37
        self.kui_ari_token_id = 38
        self.action_meld_tiles_token_id = 39

        # Chow: 0, Pong : 1, Add Kong : 2, Pei : 3, Open Kong : 4, Closed Kong : 5
        self.meld_0_base_token_id = 40
        self.meld_1_base_token_id = 46
        self.meld_2_base_token_id = 52
        self.meld_3_base_token_id = 58

        self.shanten_diff_token_id = 64
        self.shanten_token_id = 65
        self.who_token_id = 66
        self.sum_discards_token_id = 67

        self.special_token_id_list = [
            self.sep_token_id,
            self.cls_token_id,
            self.pad_token_id,
            self.mask_token_id
        ]

    def data2x(self, features, device, y):
        who = features['who']
        hand = features['hand']
        batch_size = hand.size()[0]
        cls_ids = torch.full((batch_size, 1), self.cls_token_id, dtype=torch.long, device=device)
        sep_ids = torch.full((batch_size, 1), self.sep_token_id, dtype=torch.long, device=device)

        x = torch.cat([
            cls_ids,
            hand, #14
            features['discards'][:, 0, :],
            features['discards'][:, 1, :],
            features['discards'][:, 2, :],
            features['discards'][:, 3, :], # 100(25)
            features['melds'][0][:, 0],
            features['melds'][1][:, 0],
            features['melds'][2][:, 0],
            features['melds'][3][:, 0], # 80(20)
            features['action_meld_tiles'], # 4
            features['menzen'] + self.menzen_offset,
            features['reach_state'] + self.reach_state_offset,
            features['n_reach'] + self.n_reach_offset,
            features['reach_ippatsu'] + self.reach_ippatsu_offset,
            features['doras'],
            features['dans'] + self.dans_offset,
            features['rates'] + self.rates_offset,
            features['scores'] + self.scores_offset,
            features['oya'] + self.oya_offset,
            features['n_honba'] + self.n_honba_offset,
            features['n_round'] + self.n_round_offset,
            features['sanma_or_yonma'] + self.sanma_or_yonma_offset,
            features['han_or_ton'] + self.han_or_ton_offset,
            features['aka_ari'] + self.aka_ari_offset,
            features['kui_ari'] + self.kui_ari_offset,
            features['shanten'] + self.shanten_offset,
            features['who'] + self.who_offset,
            features['sum_discards'] + self.sum_discards_offset
        ], dim=1)

        hand_length = hand.size()[1]
        discard_length = features['discards'].size()[2]
        dora_length = features['doras'].size()[1]
        pad_token_type_ids = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=device)

        token_types = torch.cat([
            pad_token_type_ids,
            torch.full((batch_size, hand_length), self.hand_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, discard_length), self.discard_0_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, discard_length), self.discard_1_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, discard_length), self.discard_2_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, discard_length), self.discard_3_token_id, dtype=torch.long, device=device),
            features['melds'][0][:, 1] + self.meld_0_base_token_id,
            features['melds'][1][:, 1] + self.meld_1_base_token_id,
            features['melds'][2][:, 1] + self.meld_2_base_token_id,
            features['melds'][3][:, 1] + self.meld_3_base_token_id,
            torch.full((batch_size, 4), self.action_meld_tiles_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.menzen_0_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.menzen_1_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.menzen_2_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.menzen_3_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.reach_state_0_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.reach_state_1_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.reach_state_2_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.reach_state_3_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.n_reach_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.reach_ippatsu_0_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.reach_ippatsu_1_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.reach_ippatsu_2_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.reach_ippatsu_3_token_id, dtype=torch.long, device=device),
            torch.full((batch_size,  dora_length), self.dora_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.dans_0_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.dans_1_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.dans_2_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.dans_3_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.rates_0_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.rates_1_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.rates_2_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.rates_3_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.scores_0_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.scores_1_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.scores_2_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.scores_3_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.oya_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.n_honba_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.n_round_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.sanma_or_yonma_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.han_or_ton_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.aka_ari_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.kui_ari_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.shanten_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.who_token_id, dtype=torch.long, device=device),
            torch.full((batch_size, 1), self.sum_discards_token_id, dtype=torch.long, device=device)
        ], dim=1)


        cls_tokens = torch.tensor([self.tgt_cls_token_id] * batch_size, dtype=torch.long, device=device).reshape((batch_size, 1))
        tgt_ids = torch.cat([cls_tokens, y[:, :-1]], axis=1)
        tgt_ids[tgt_ids == -100] = self.tgt_pad_token_id

        return x, token_types, tgt_ids


    def forward(self, x, token_type_ids, y):
        y_embeddings = self.tgt_embeddings(y)
        x_embeddings = self.symbol_embeddings(x)
        x_embeddings += self.token_type_embeddings(token_type_ids)

        x_embeddings = self.layer_norm(x_embeddings)
        x_embeddings = self.dropout(x_embeddings)
        return x_embeddings, y_embeddings


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def accuracy_fct(logits, y, n_classes):
    _, pred = torch.max(logits, 1)
    corrects = pred == y
    enableds = (y != -100)
    # print(f'enableds:{(y != -100)}')
    # print(f'pred:{pred.reshape((-1, 43))}')
    return torch.tensor(
        corrects.sum().item() / enableds.sum().item(),
        dtype=torch.float, device=y.device
    )


class MahjongModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = MahjongEmbeddings(config)
        self.transformer = nn.Transformer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward']
        )
        self.pos_encoder = PositionalEncoding(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['num_outputs'])
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)

        self.src_mask = None
        self.tgt_mask = None


    def forward(self, x_features, y):
        x, token_type_ids, tgt_ids = self.embeddings.data2x(x_features, y.device, y)
        src_emb, tgt_emb = self.embeddings(x, token_type_ids, tgt_ids)
        src_emb = self.pos_encoder(src_emb * math.sqrt(self.config['d_model']))
        tgt_emb = self.pos_encoder(tgt_emb * math.sqrt(self.config['d_model']))

        if self.tgt_mask is None:
            mask = self.transformer.generate_square_subsequent_mask(y.shape[1]).to(y.device)
            self.tgt_mask = mask

        # if self.src_mask is None:
        #     mask = self.generate_square_subsequent_mask(embedding_output.shape[1]).to(y.device)
        #     self.src_mask = mask
        # src_mask = self.generate_src_mask(x)

        transformer_output = self.transformer(
            src_emb.permute(1, 0, 2), # (batch size, seq len, hidden size) -> (seq len, batch size, hidden size)
            tgt_emb.permute(1, 0, 2), #
            # self.src_mask,
            tgt_mask=self.tgt_mask
        )
        logits = self.head(
            transformer_output.permute(1, 0, 2) # (tgt len, batch size, hidden size) -> (batch size, tgt len, hidden size)
        )

        # print(f'logits:{logits.shape}, y:{y}')

        y[(y != -100) * (y != 39)] -= 1
        y[y == 39] = 37


        # print(f'self.tgt_mask:{self.tgt_mask}, tgt_ids:{tgt_ids}, y:{y}')

        # print(f'logits:{logits.shape}, y:{y}')

        loss = self.loss_fct(
            logits.view(-1, self.config['num_outputs']),
            y.reshape(-1)
        )
        accuracy = accuracy_fct(
            logits.view(-1, self.config['num_outputs']),
            y.reshape(-1),
            self.config['num_outputs']
        )

        return loss, accuracy

    def predict(self, x_features, y):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'


        # print(f'x_features: {x_features}')
        # print(f'y: {y}')

        l = len(y)
        y_tensor = torch.full(((14 + 1) * 3, ), -100, dtype=torch.long, device=device)
        y_tensor[:l] = torch.tensor(y, dtype=torch.long, device=device)
        y_tensor = y_tensor.unsqueeze(0)

        x, token_type_ids, tgt_ids = self.embeddings.data2x(x_features, device, y_tensor)
        # print(f'tgt_ids:{tgt_ids}')

        src_emb, tgt_emb = self.embeddings(x, token_type_ids, tgt_ids)
        src_emb = self.pos_encoder(src_emb * math.sqrt(self.config['d_model']))
        tgt_emb = self.pos_encoder(tgt_emb * math.sqrt(self.config['d_model']))

        if self.tgt_mask is None:
            mask = self.transformer.generate_square_subsequent_mask(y_tensor.shape[1]).to(device)
            self.tgt_mask = mask

        transformer_output = self.transformer(
            src_emb.permute(1, 0, 2), # (batch size, seq len, hidden size) -> (seq len, batch size, hidden size)
            tgt_emb.permute(1, 0, 2), #
            tgt_mask=self.tgt_mask
        )
        logits = self.head(
            transformer_output.permute(1, 0, 2) # (tgt len, batch size, hidden size) -> (batch size, tgt len, hidden size)
        )
        logits = logits[0][len(y)]
        prob = self.softmax(logits.unsqueeze(0))
        log_p = torch.log(prob)[0]
        return log_p
