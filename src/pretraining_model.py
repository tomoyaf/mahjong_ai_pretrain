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


def get_model(enable_model_name):
    config = BertConfig()
    # tile(37), menzen(2), reach_state(2), n_reach(3),
    # reach_ippatsu(2), dans(21), rates(19), oya(4),
    # scores(13), n_honba(3), n_round(12), sanma_or_yonma(2),
    # han_or_ton(2), aka_ari(2), kui_ari(2), special_token(4)
    config.vocab_size = 37 + 2 + 2 + 3 + 2 + 21 + 19 + 4 + 13 + 3 + 12 + 2 + 2 + 2 + 2 + 4
    # config.hidden_size = 1024
    config.hidden_size = 768
    config.num_attention_heads = 12
    # config.num_attention_heads = 16
    # config.num_hidden_layers = 2
    config.max_position_embeddings = 260

    discard_config = BertConfig()
    discard_config.vocab_size = config.vocab_size
    discard_config.hidden_size = config.hidden_size
    discard_config.num_attention_heads = config.num_attention_heads
    discard_config.max_position_embeddings = config.max_position_embeddings
    # discard_config.num_hidden_layers = 2
    # discard_config.num_hidden_layers = 24
    discard_config.num_hidden_layers = 12

    reach_config = BertConfig()
    reach_config.vocab_size = config.vocab_size
    reach_config.hidden_size = config.hidden_size
    reach_config.num_attention_heads = config.num_attention_heads
    reach_config.max_position_embeddings = config.max_position_embeddings
    reach_config.num_hidden_layers = 6

    chow_config = BertConfig()
    chow_config.vocab_size = config.vocab_size
    chow_config.hidden_size = config.hidden_size
    chow_config.num_attention_heads = config.num_attention_heads
    chow_config.max_position_embeddings = config.max_position_embeddings
    chow_config.num_hidden_layers = 6

    pong_config = BertConfig()
    pong_config.vocab_size = config.vocab_size
    pong_config.hidden_size = config.hidden_size
    pong_config.num_attention_heads = config.num_attention_heads
    pong_config.max_position_embeddings = config.max_position_embeddings
    pong_config.num_hidden_layers = 6

    kong_config = BertConfig()
    kong_config.vocab_size = config.vocab_size
    kong_config.hidden_size = config.hidden_size
    kong_config.num_attention_heads = config.num_attention_heads
    kong_config.max_position_embeddings = config.max_position_embeddings
    kong_config.num_hidden_layers = 6


    return MahjongModelForPreTraining(
        config,
        discard_config,
        reach_config,
        chow_config,
        pong_config,
        kong_config,
        enable_model_name
    )


def get_optimizer(model, lr=1e-4, weight_decay=0.01, n_epochs=10, n_warmup_steps=1e4, n_training_steps=1e5):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=n_warmup_steps,
        num_training_steps=n_training_steps
    )
    return optimizer, lr_scheduler


def get_loaders(batch_size, model_name):
    # path_list = glob(f'./pickle/*/paifu_2018_*.pickle')
    path_list = glob(f'./pickle/{model_name}/paifu_2018_*.pickle')
    np.random.shuffle(path_list)
    # path_list = path_list[:20000]

    data_size = len(path_list)
    train_size = int(data_size * 0.8)
    val_size = data_size - train_size

    print(f'Data size : {data_size}, Train size : {train_size}, Val size : {val_size}')

    train_path_list = path_list[:train_size]
    val_path_list = path_list[-val_size:]

    train_dataset = PaifuDataset(train_path_list)
    val_dataset = PaifuDataset(val_path_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


class MahjongEmbeddings(nn.Module):
    def __init__(self, config, n_token_type=31):
        super(MahjongEmbeddings, self).__init__()
        print(config)
        self.config = config
        self.symbol_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
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

        # Token type id
        self.hand_token_id = 0
        self.discard_0_token_id = 1
        self.discard_1_token_id = 2
        self.discard_2_token_id = 3
        self.discard_3_token_id = 4
        self.menzen_token_id = 5
        self.reach_state_token_id = 6
        self.n_reach_token_id = 7
        self.reach_ippatsu_token_id = 8
        self.dora_token_id = 9
        self.dans_token_id = 10
        self.rates_token_id = 11
        self.scores_token_id = 12
        self.oya_token_id = 13
        self.n_honba_token_id = 14
        self.n_round_token_id = 15
        self.sanma_or_yonma_token_id = 16
        self.han_or_ton_token_id = 17
        self.aka_ari_token_id = 18
        self.kui_ari_token_id = 19
        self.action_meld_tiles_token_id = 20

        # Chow: 0, Pong : 1, Add Kong : 2, Pei : 3, Open Kong : 4, Closed Kong : 5
        self.meld_base_token_id = 21

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

        x = torch.cat([
            cls_ids,
            hand, #14
            sep_ids,
            features['discards'][:, 0, :],
            sep_ids,
            features['discards'][:, 1, :],
            sep_ids,
            features['discards'][:, 2, :],
            sep_ids,
            features['discards'][:, 3, :], # 100(25)
            sep_ids,
            features['melds'][0][:, 0],
            sep_ids,
            features['melds'][1][:, 0],
            sep_ids,
            features['melds'][2][:, 0],
            sep_ids,
            features['melds'][3][:, 0], # 80(20)
            sep_ids,
            features['action_meld_tiles'], # 4
            sep_ids, # 49
            features['menzen'] + self.menzen_offset,
            sep_ids,
            features['reach_state'] + self.reach_state_offset,
            sep_ids,
            features['n_reach'] + self.n_reach_offset,
            sep_ids,
            features['reach_ippatsu'] + self.reach_ippatsu_offset,
            sep_ids,
            features['doras'],
            sep_ids,
            features['dans'] + self.dans_offset,
            sep_ids,
            features['rates'] + self.rates_offset,
            sep_ids,
            features['scores'] + self.scores_offset,
            sep_ids,
            features['oya'] + self.oya_offset,
            sep_ids,
            features['n_honba'] + self.n_honba_offset,
            sep_ids,
            features['n_round'] + self.n_round_offset,
            sep_ids,
            features['sanma_or_yonma'] + self.sanma_or_yonma_offset,
            sep_ids,
            features['han_or_ton'] + self.han_or_ton_offset,
            sep_ids,
            features['aka_ari'] + self.aka_ari_offset,
            sep_ids,
            features['kui_ari'] + self.kui_ari_offset,
        ], dim=1)

        hand_length = hand.size()[1]
        discard_length = features['discards'].size()[2]
        dora_length = features['doras'].size()[1]
        pad_token_type_ids = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=device)

        token_types = torch.cat([
            pad_token_type_ids,
            torch.full((batch_size, hand_length), self.hand_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, discard_length), self.discard_0_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, discard_length), self.discard_1_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, discard_length), self.discard_2_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, discard_length), self.discard_3_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            features['melds'][0][:, 1] + self.meld_base_token_id,
            pad_token_type_ids,
            features['melds'][1][:, 1] + self.meld_base_token_id,
            pad_token_type_ids,
            features['melds'][2][:, 1] + self.meld_base_token_id,
            pad_token_type_ids,
            features['melds'][3][:, 1] + self.meld_base_token_id,
            pad_token_type_ids,
            torch.full((batch_size, 4), self.action_meld_tiles_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 4), self.menzen_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 4), self.reach_state_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 1), self.n_reach_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 4), self.reach_ippatsu_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size,  dora_length), self.dora_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 4), self.dans_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 4), self.rates_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 4), self.scores_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 1), self.oya_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 1), self.n_honba_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 1), self.n_round_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 1), self.sanma_or_yonma_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 1), self.han_or_ton_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 1), self.aka_ari_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, 1), self.kui_ari_token_id, dtype=torch.long, device=device)
        ], dim=1)

        seq_len = x.shape[1]
        pos_ids = torch.arange(
            self.config.max_position_embeddings,
            dtype=torch.long,
            device=catalyst.utils.get_device()
        )
        pos_ids = torch.cat(
            [pos_ids[:seq_len]] * batch_size
        )
        pos_ids = pos_ids.reshape((batch_size, seq_len))
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
    def __init__(self, config, output_dim):
        super().__init__()
        self.dense0 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense1 = nn.Linear(config.hidden_size, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dense1.bias = self.bias

        self.act_fct = torch.tanh
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        h = self.dense0(hidden_states[:, 0])
        h = self.act_fct(h)
        h = self.layer_norm(h)
        h = self.dense1(h)
        return h

class DiscardHead(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.config = config
        self.n_hands = 14
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        h = self.dense(hidden_states[:, 1:self.n_hands+1, :])
        return h


class MahjongModelForPreTraining(nn.Module):
    def __init__(
        self, config,
        discard_config, reach_config, chow_config, pong_config, kong_config,
        enabled_model_name
    ):
        super().__init__()
        self.config = config
        self.enabled_model_name = enabled_model_name
        self.discard_config = discard_config
        self.reach_config = reach_config
        self.chow_config = chow_config
        self.pong_config = pong_config
        self.kong_config = kong_config
        self.embeddings = MahjongEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.discard_bert_encoder = BertEncoder(discard_config)
        self.reach_bert_encoder = BertEncoder(reach_config)
        self.chow_bert_encoder = BertEncoder(chow_config)
        self.pong_bert_encoder = BertEncoder(pong_config)
        self.kong_bert_encoder = BertEncoder(kong_config)
        # self.discard_output_dim = 37
        self.discard_output_dim = 14
        self.reach_output_dim = 2
        self.chow_output_dim = 2
        self.pong_output_dim = 2
        self.kong_output_dim = 2
        # self.discard_head = BertHead(config, self.discard_output_dim)
        self.discard_head = DiscardHead(config, self.discard_output_dim)
        self.reach_head = BertHead(config, self.reach_output_dim)
        self.chow_head = BertHead(config, self.chow_output_dim)
        self.pong_head = BertHead(config, self.pong_output_dim)
        self.kong_head = BertHead(config, self.kong_output_dim)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.discard_loss_fct = torch.nn.CrossEntropyLoss()
        # self.discard_loss_fct = torch.nn.BCEWithLogitsLoss()
        self.mlm_probability = 0.15

    def forward(
        self,
        x_features,
        y,
        attention_mask=None,
        token_type_id=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        x, token_type_ids, pos_ids = self.embeddings.data2x(x_features, y.device)
        embedding_output = self.embeddings(x, token_type_ids, pos_ids)

        if self.enabled_model_name == 'discard':
            bert_outputs = self.discard_bert_encoder(embedding_output)
            last_hidden_state = bert_outputs[0]
            logits = self.discard_head(last_hidden_state)
            loss = self.discard_loss_fct(
                logits.view(-1, self.discard_output_dim),
                y[:, 0].reshape(-1)
            )
            accuracy = self.accuracy_fct(
                logits.view(-1, self.discard_output_dim),
                y[:, 0].reshape(-1),
                self.discard_output_dim
            )
        elif self.enabled_model_name == 'reach':
            bert_outputs = self.reach_bert_encoder(embedding_output)
            last_hidden_state = bert_outputs[0]
            logits = self.reach_head(last_hidden_state)
            loss = self.loss_fct(
                logits.view(-1, self.reach_output_dim),
                y[:, 1].reshape(-1)
            )
            accuracy = self.accuracy_fct(
                logits.view(-1, self.reach_output_dim),
                y[:, 1].reshape(-1),
                self.reach_output_dim
            )
        elif self.enabled_model_name == 'chow':
            bert_outputs = self.chow_bert_encoder(embedding_output)
            last_hidden_state = bert_outputs[0]
            logits = self.chow_head(last_hidden_state)
            loss = self.loss_fct(
                logits.view(-1, self.chow_output_dim),
                y[:, 2].reshape(-1)
            )
            accuracy = self.accuracy_fct(
                logits.view(-1, self.chow_output_dim),
                y[:, 2].reshape(-1),
                self.chow_output_dim
            )
        elif self.enabled_model_name == 'pong':
            bert_outputs = self.pong_bert_encoder(embedding_output)
            last_hidden_state = bert_outputs[0]
            logits = self.pong_head(last_hidden_state)
            loss = self.loss_fct(
                logits.view(-1, self.pong_output_dim),
                y[:, 3].reshape(-1)
            )
            accuracy = self.accuracy_fct(
                logits.view(-1, self.pong_output_dim),
                y[:, 3].reshape(-1),
                self.pong_output_dim
            )
        elif self.enabled_model_name == 'kong':
            bert_outputs = self.kong_bert_encoder(embedding_output)
            last_hidden_state = bert_outputs[0]
            logits = self.kong_head(last_hidden_state)
            loss = self.loss_fct(
                logits.view(-1, self.kong_output_dim),
                y[:, 4].reshape(-1)
            )
            accuracy = self.accuracy_fct(
                logits.view(-1, self.kong_output_dim),
                y[:, 4].reshape(-1),
                self.kong_output_dim
            )

        return loss, accuracy


    def accuracy_fct(self, logits, y, n_classes):
        _, pred = torch.max(logits, 1)
        corrects = pred == y
        enableds = (y != -100)
        return torch.tensor(corrects.sum().item() / enableds.sum().item(), dtype=torch.float, device=y.device)

    def calc_n_pai_type(self, hand):
        one_hot = torch.nn.functional.one_hot(
            hand,
            num_classes=self.discard_output_dim + 1
        )
        one_hot = one_hot[:, :, 1:]
        one_hot = torch.sum(one_hot, axis=1)
        one_hot[one_hot > 1] = 1
        n_pai_type = torch.sum(one_hot, axis=1, dtype=torch.float).mean()
        return n_pai_type



    def f_score_fct(self, logits, y):
        _, pred = torch.max(logits, 1)
        true_positive = (pred == 1) & (y == 1)
        false_positive = (pred == 1) & (y == 0)
        false_negative = (pred == 0) & (y == 1)

        tp = true_positive.sum().item()
        fp = false_positive.sum().item()
        fn = false_negative.sum().item()

        # print(f'tp:{tp}, fp:{fp}, fn:{fn}')

        if tp + fp == 0 or tp + fn == 0:
            return -100.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision == 0 and recall == 0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)


    def mask_discard_by_hand(self, discard_logits, hand):
        one_hot = torch.nn.functional.one_hot(hand, num_classes=self.discard_output_dim + 1)
        one_hot = one_hot[:, :, 1:]
        one_hot = torch.sum(one_hot, axis=1)
        discard_logits[one_hot < 1] = -1e10

        return discard_logits


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
