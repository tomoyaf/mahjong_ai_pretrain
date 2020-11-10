import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertLayer, BertConfig
from glob import glob
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import catalyst

from pretraining_data import PaifuDataset


def get_model(enable_model_name, is_pretraining, pretrained_path):
    # tile(37), menzen(2), reach_state(2), n_reach(3),
    # reach_ippatsu(2), dans(21), rates(19), oya(4),
    # scores(13), n_honba(3), n_round(12), sanma_or_yonma(2),
    # han_or_ton(2), aka_ari(2), kui_ari(2), special_token(4)
    # vocab_size = 37 + 2 + 2 + 3 + 2 + 21 + 19 + 4 + 13 + 3 + 12 + 2 + 2 + 2 + 2 + 4 + 2 + 4 + 6 + 8 # 130 + shanten_diff(2) + who(4) + sum_discards(6) + shanten(8)
    vocab_size = 37 + 2 + 2 + 3 + 2 + 21 + 19 + 4 + 13 + 3 + 12 + 2 + 2 + 2 + 2 + 4 + 4 + 6 + 8 # 130 + who(4) + sum_discards(6) + shanten(8)
    # hidden_size = 1024
    # num_attention_heads = 16
    hidden_size = 768
    num_attention_heads = 12
    max_position_embeddings = 239 # base + pad(1) + who(1) + pad(1) + sum_discards(1) + pad(1) + shanten(1)
    # max_position_embeddings = 281 # 260 + pad(1) + shanten_diff(14) + pad(1) + who(1) + pad(1) + sum_discards(1) + pad(1) + shanten(1)

    if is_pretraining:
        config = BertConfig()
        config.vocab_size = vocab_size
        config.hidden_size = hidden_size
        config.num_attention_heads = num_attention_heads
        config.max_position_embeddings = max_position_embeddings
        config.num_hidden_layers = 12
        return MahjongPretrainingModel(config)

    model = None
    if enable_model_name == 'discard':
        discard_config = BertConfig()
        discard_config.vocab_size = vocab_size
        discard_config.hidden_size = hidden_size
        discard_config.num_attention_heads = num_attention_heads
        discard_config.max_position_embeddings = max_position_embeddings
        discard_config.num_hidden_layers = 12
        # discard_config.num_hidden_layers = 24
        # discard_config.num_hidden_layers = 12
        model = MahjongDiscardModel(discard_config)
    elif enable_model_name == 'reach':
        reach_config = BertConfig()
        reach_config.vocab_size = vocab_size
        reach_config.hidden_size = hidden_size
        reach_config.num_attention_heads = num_attention_heads
        reach_config.max_position_embeddings = max_position_embeddings
        reach_config.num_hidden_layers = 24
        model = MahjongReachChowPongKongModel(reach_config)
    elif enable_model_name == 'chow':
        chow_config = BertConfig()
        chow_config.vocab_size = vocab_size
        chow_config.hidden_size = hidden_size
        chow_config.num_attention_heads = num_attention_heads
        chow_config.max_position_embeddings = max_position_embeddings
        chow_config.num_hidden_layers = 24
        model = MahjongReachChowPongKongModel(chow_config)
    elif enable_model_name == 'pong':
        pong_config = BertConfig()
        pong_config.vocab_size = vocab_size
        pong_config.hidden_size = hidden_size
        pong_config.num_attention_heads = num_attention_heads
        pong_config.max_position_embeddings = max_position_embeddings
        pong_config.num_hidden_layers = 24
        model = MahjongReachChowPongKongModel(pong_config)
    elif enable_model_name == 'kong':
        kong_config = BertConfig()
        kong_config.vocab_size = vocab_size
        kong_config.hidden_size = hidden_size
        kong_config.num_attention_heads = num_attention_heads
        kong_config.max_position_embeddings = max_position_embeddings
        kong_config.num_hidden_layers = 24
        model = MahjongReachChowPongKongModel(kong_config)


    if pretrained_path != '':
        checkpoint = torch.load(pretrained_path, map_location=catalyst.utils.get_device())
        # print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False
        )

    return model


def get_optimizer(model, lr=1e-4, weight_decay=0.01, n_epochs=10, n_warmup_steps=1e4, n_training_steps=4e5):
    print(f'lr:{lr}, weight_decay:{weight_decay}, n_training_steps:{n_training_steps}, n_warmup_steps:{n_warmup_steps}')
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=n_warmup_steps,
        num_training_steps=n_training_steps
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


def get_loaders(batch_size, model_name, max_data_size, is_pretraining, n_max=100):
    train_path_list = []
    val_path_list = []
    test_path_list = []
    data_size = 0
    train_size = 0
    val_size = 0
    test_size = 0
    base_path = './preprocessed'

    if is_pretraining:
        path_list_list = [
            glob(f'{base_path}/discard/*'),
            glob(f'{base_path}/reach/*'),
            glob(f'{base_path}/chow/*'),
            glob(f'{base_path}/pong/*'),
            glob(f'{base_path}/kong/*')
        ]
        # for path_list in path_list_list:
        #     train_path_list_i, val_path_list_i, test_path_list_i, data_size_i, train_size_i, val_size_i, test_size_i = process_path_list(path_list, max_data_size // 5)
        #     train_path_list += train_path_list_i
        #     val_path_list += val_path_list_i
        #     test_path_list += test_path_list_i
        #     data_size += data_size_i
        #     train_size += train_size_i
        #     val_size += val_size_i
        #     test_size += test_size_i
        #     print(f'Partial Data size : {data_size_i}, Partial Train Size : {train_size_i}, Partial Val Size : {val_size_i}, Partial Val Size : {test_size_i}')
    else:
        path_list = glob(f'{base_path}/{model_name}/*')
        train_path_list, val_path_list, test_path_list, data_size, train_size, val_size, test_size = process_path_list(path_list, max_data_size, n_max)

    # np.random.shuffle(train_path_list)
    # np.random.shuffle(val_path_list)
    # np.random.shuffle(test_path_list)

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
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.n_position_embeddings = 25 + 20 + 1 # discard : 25, meld : 20, pad
        self.position_embeddings = nn.Embedding(
            self.n_position_embeddings,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            n_token_type,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Special token id
        self.sep_token_id = 127
        self.cls_token_id = 128
        self.mask_token_id = 129
        self.pad_token_id = config.pad_token_id

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

    def data2x(self, features, device):
        who = features['who']
        hand = features['hand']
        batch_size = hand.size()[0]
        cls_ids = torch.full((batch_size, 1), self.cls_token_id, dtype=torch.long, device=device)
        sep_ids = torch.full((batch_size, 1), self.sep_token_id, dtype=torch.long, device=device)

        # features['shanten_diff'] += self.shanten_diff_offset
        # features['shanten_diff'][features['shanten_diff'] == self.shanten_diff_offset - 1] = self.pad_token_id

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

        meld_length = 20
        pos_arange = torch.arange(self.n_position_embeddings - 1 , dtype=torch.long, device=device) + 1
        discard_pos_ids = torch.cat([pos_arange[:discard_length]] * batch_size).reshape((batch_size, discard_length))
        meld_pos_ids = torch.cat([pos_arange[discard_length : discard_length + meld_length]] * batch_size).reshape((batch_size, meld_length))
        pos_ids = torch.zeros((x.size()[0], x.size()[1]), dtype=torch.long, device=device)

        for i in range(4):
            discard_start_idx = (1 + hand_length) + discard_length * i
            discard_end_idx = discard_start_idx + discard_length
            pos_ids[:, discard_start_idx : discard_end_idx] = discard_pos_ids

            meld_start_idx = (1 + hand_length) + discard_length * 4 + meld_length * i
            meld_end_idx = meld_start_idx + meld_length
            pos_ids[:, meld_start_idx : meld_end_idx] = meld_pos_ids

        return x, token_types, pos_ids


    def forward(self, x, token_type_ids, pos_ids):
        embeddings = self.symbol_embeddings(x)
        embeddings += self.token_type_embeddings(token_type_ids)
        embeddings += self.position_embeddings(pos_ids)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states
            )
            # layer_outputs = layer_module(
            #     hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            # )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.act_fct = torch.tanh
        self.act_fct = F.gelu
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        h = self.dense(hidden_states)
        h = self.act_fct(h)
        h = self.layer_norm(h)
        return h

class MLMHead(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dense.bias = self.bias

    def forward(self, h):
        h = self.dense(h)
        return h


class ReachPongChowKongHead(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dense.bias = self.bias

    def forward(self, h):
        h = self.dense(h)
        return h

class DiscardHead(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.config = config
        self.n_hands = 14
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        h = self.dense(hidden_states[:, 1:self.n_hands+1])
        return h


def accuracy_fct(logits, y, n_classes):
    _, pred = torch.max(logits, 1)
    corrects = pred == y
    enableds = (y != -100)
    return torch.tensor(
        corrects.sum().item() / enableds.sum().item(),
        dtype=torch.float, device=y.device
    )


class MahjongPretrainingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config.vocab_size

        self.embeddings = MahjongEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert_head = BertHead(config)
        self.pretrained_model_head = MLMHead(config, config.vocab_size)

        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.device = catalyst.utils.get_device()
        self.mlm_probability = 0.15


    def forward(self, x_features, _):
        x, token_type_ids, pos_ids = self.embeddings.data2x(x_features, self.device)
        x, y = self.mask_tokens(
            x,
            mlm_probability=self.mlm_probability,
            special_token_id_list=self.embeddings.special_token_id_list,
            mask_token_id=self.embeddings.mask_token_id,
            device=self.device
        )
        embedding_output = self.embeddings(x, token_type_ids, pos_ids)

        bert_outputs = self.bert_encoder(embedding_output)
        sequence_output = bert_outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.bert_head(sequence_output)
        logits = self.pretrained_model_head(sequence_output)
        loss = self.loss_fct(
            logits.view(-1, self.config.vocab_size),
            y.view(-1)
        )
        accuracy = accuracy_fct(
            logits.view(-1, self.config.vocab_size),
            y.view(-1),
            self.config.vocab_size
        )

        return loss, accuracy


    def mask_tokens(
        self,
        inputs,
        mlm_probability,
        special_token_id_list,
        mask_token_id,
        device
    ):
        labels = inputs.clone()

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability, device=device)

        for sp_token_id in special_token_id_list:
            msk = labels.eq(sp_token_id)
            probability_matrix.masked_fill_(msk, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8, device=device)
        ).bool() & masked_indices
        inputs[indices_replaced] = mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5, device=device)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            self.output_dim,
            labels.shape,
            dtype=torch.long,
            device=device
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class MahjongDiscardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = MahjongEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.output_dim = 14
        self.bert_head = BertHead(config)
        self.discard_head = DiscardHead(config, self.output_dim)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, x_features, y):
        x, token_type_ids, pos_ids = self.embeddings.data2x(x_features, y.device)
        embedding_output = self.embeddings(x, token_type_ids, pos_ids)

        # print(f'x{x.shape}, token:{token_type_ids.shape}, pos:{pos_ids.shape}')
        # print(f'x{x}, token:{token_type_ids}, pos:{pos_ids}')

        bert_outputs = self.bert_encoder(embedding_output)
        last_hidden_state = bert_outputs[0]
        last_hidden_state = self.bert_head(last_hidden_state)
        logits = self.discard_head(last_hidden_state)
        loss = self.loss_fct(
            logits.view(-1, self.output_dim),
            y.reshape(-1)
        )
        accuracy = accuracy_fct(
            logits.view(-1, self.output_dim),
            y.reshape(-1),
            self.output_dim
        )

        return loss, accuracy


class MahjongReachChowPongKongModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = MahjongEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.output_dim = 2
        self.bert_head = BertHead(config)
        self.reach_chow_pong_kong_head = ReachChowPongKongHead(config, self.output_dim)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, x_features, y):
        x, token_type_ids, pos_ids = self.embeddings.data2x(x_features, y.device)
        embedding_output = self.embeddings(x, token_type_ids, pos_ids)

        bert_outputs = self.bert_encoder(embedding_output)
        last_hidden_state = bert_outputs[0]
        last_hidden_state = self.bert_head(last_hidden_state)
        logits = self.reach_chow_pong_kong_head(last_hidden_state)
        loss = self.loss_fct(
            logits.view(-1, self.output_dim),
            y.reshape(-1)
        )
        accuracy = accuracy_fct(
            logits.view(-1, self.output_dim),
            y.reshape(-1),
            self.output_dim
        )

        return loss, accuracy
