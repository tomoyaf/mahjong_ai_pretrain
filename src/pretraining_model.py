import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertLayer, BertConfig
from glob import glob
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from pretraining_data import PaifuDataset


def get_model():
    config = BertConfig()
    config.vocab_size = 37 + 5
    # config.hidden_size = 120 # 768
    config.hidden_size = 72
    # config.num_attention_heads = 12
    config.num_hidden_layers = 5
    # config.max_position_embeddings = 512
    return MahjongModelForPreTraining(config)


def get_optimizer(model, lr=1e-4, weight_decay=0.01, n_epochs=10, n_warmup_steps=1e4, n_training_steps=1e5):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=n_warmup_steps,
        num_training_steps=n_training_steps
    )
    return optimizer, lr_scheduler


def get_loaders(batch_size=8, num_workers=8, model_types=['discard', 'reach', 'chow', 'pong', 'kong']):
    path_list_dict = {}
    whole_path_list = []
    for model_type in model_types:
        path_list = glob(f'./pickle/{model_type}/paifu_2018_*.pickle')
        path_list_dict[model_type] = path_list[:10000]
        whole_path_list = [*whole_path_list, *path_list_dict[model_type]]

    np.random.shuffle(whole_path_list)

    data_size = len(whole_path_list[model_type] )
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
    def __init__(self, config, n_token_type=13):
        super(MahjongEmbeddings, self).__init__()
        print(config)
        self.tile_embeddings = nn.Embedding(
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

        self.sep_token_id = 37
        self.cls_token_id = 38
        self.mask_token_id = 39
        self.pad_token_id = config.pad_token_id

        # self.hand_tile_token_id = 1
        # self.known_hand_tile_token_id = 2
        self.discards_tile_token_id = 3

        self.who_0_hand_token_id = 4
        self.who_1_hand_token_id = 5
        self.who_2_hand_token_id = 4
        self.who_3_hand_token_id = 7

        self.who_0_discard_token_id = 8
        self.who_1_discard_token_id = 9
        self.who_2_discard_token_id = 10
        self.who_3_discard_token_id = 11

        self.dora_token_id = 12

        # Add Chow: 0, Pong : 1, Kong : 2
        # Result Chow: 13, Pong: 14, Kong: 15
        self.meld_base_token_id = 13

        self.special_token_id_list = [
            self.sep_token_id,
            self.cls_token_id,
            self.pad_token_id,
            self.mask_token_id
        ]



    def data2x(self, features, device):
        batch_size = hand.size()[0]
        cls_ids = torch.full((batch_size, 1), self.cls_token_id, dtype=torch.long, device=device)
        sep_ids = torch.full((batch_size, 1), self.sep_token_id, dtype=torch.long, device=device)
        who = features['action']['who']
        hand = features['hands'][:, who, :]
        x = torch.cat([
            cls_ids,
            hand,
            sep_ids,
            features['discards'][:, 0, :],
            sep_ids,
            features['discards'][:, 1, :],
            sep_ids,
            features['discards'][:, 2, :],
            sep_ids,
            features['discards'][:, 3, :],
            sep_ids,
            features['melds'][0][:, 0],
            sep_ids,
            features['melds'][1][:, 0],
            sep_ids,
            features['melds'][2][:, 0],
            sep_ids,
            features['melds'][3][:, 0],
            sep_ids,
            features['doras']
        ], dim=1)

        hand_length = hand.size()[2]
        discard_length = features['discards'].size()[2]
        dora_length = features['doras'].size()[1]
        pad_token_type_ids = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=device)

        token_types = torch.cat([
            pad_token_type_ids,
            torch.full((batch_size, hand_length), self.who_0_hand_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            pad_token_type_ids,
            torch.full((batch_size, hand_length), self.who_1_hand_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            pad_token_type_ids,
            torch.full((batch_size, hand_length), self.who_2_hand_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            pad_token_type_ids,
            torch.full((batch_size, hand_length), self.who_3_hand_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, discard_length), self.who_0_discard_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, discard_length), self.who_1_discard_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, discard_length), self.who_2_discard_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            torch.full((batch_size, discard_length), self.who_3_discard_token_id, dtype=torch.long, device=device),
            pad_token_type_ids,
            features['melds'][0][:, 1] + self.meld_base_token_id,
            pad_token_type_ids,
            features['melds'][1][:, 1] + self.meld_base_token_id,
            pad_token_type_ids,
            features['melds'][2][:, 1] + self.meld_base_token_id,
            pad_token_type_ids,
            features['melds'][3][:, 1] + self.meld_base_token_id,
            pad_token_type_ids,
            torch.full((batch_size,  dora_length), self.dora_token_id, dtype=torch.long, device=device)
        ], dim=1)

        discard_pos_ids = torch.arange(discard_length + 1, dtype=torch.long, device=device)
        discard_pos_ids = torch.cat([discard_pos_ids] * batch_size).reshape((batch_size, discard_length + 1))
        pos_ids = torch.zeros((x.size()[0], x.size()[1]), dtype=torch.long, device=device)

        # hand, discard, meld, dora
        discard_start_idx = (hand_length + 2) * 4 + (discard_length + 1) * 0
        discard_end_idx = discard_start_idx + (discard_length + 1)
        pos_ids[:, discard_start_idx : discard_end_idx] = discard_pos_ids

        discard_start_idx = (hand_length + 2) * 4 + (discard_length + 1) * 1
        discard_end_idx = discard_start_idx + (discard_length + 1)
        pos_ids[:, discard_start_idx : discard_end_idx] = discard_pos_ids

        discard_start_idx = (hand_length + 2) * 4 + (discard_length + 1) * 2
        discard_end_idx = discard_start_idx + (discard_length + 1)
        pos_ids[:, discard_start_idx : discard_end_idx] = discard_pos_ids

        discard_start_idx = (hand_length + 2) * 4 + (discard_length + 1) * 3
        discard_end_idx = discard_start_idx + (discard_length + 1)
        pos_ids[:, discard_start_idx : discard_end_idx] = discard_pos_ids

        return x, token_types, pos_ids


    def forward(self, x, token_type_ids, pos_ids):
        # device = x.device
        # batch_size = x.size()[0]
        # sep_embeddings = self.special_token_emb(self.sep_token_id, batch_size, device)

        # hand_tile_embeddings = self.hand_tile_emb(x, sep_embeddings, batch_size, device)
        # discards_tile_embeddings = self.discards_tile_emb(x, sep_embeddings, batch_size, device)
        # melds_tile_embeddings= self.meld_emb(x)

        # embeddings = torch.cat([
        #     hand_tile_embeddings,
        #     sep_embeddings,
        #     discards_tile_embeddings,
        #     sep_embeddings,
        #     melds_tile_embeddings
        # ], 1)
        # embeddings = hand_tile_embeddings

        embeddings = self.tile_embeddings(x)

        # batch_size = x.size()[0]
        # seq_length = x.size()[1]
        # device = x.device
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        # position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))
        # position_embeddings = self.position_embeddings(position_ids)
        # embeddings += position_embeddings

        embeddings += self.token_type_embeddings(token_type_ids)
        embeddings += self.position_embeddings(pos_ids)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


    def pos_emb(self, l, batch_size, device):
        ids = torch.arange(l, dtype=torch.long, device=device)
        ids = torch.cat([ids] * batch_size).reshape((batch_size, l))
        return self.position_embeddings(ids)


    def token_type_emb(self, l, type_id, batch_size, device):
        ids = torch.full((batch_size, l), fill_value=type_id, dtype=torch.long, device=device)
        return self.token_type_embeddings(ids)


    def special_token_emb(self, token_id, batch_size, device):
        return self.tile_embeddings(
            torch.full(
                (batch_size, 1),
                fill_value=token_id,
                dtype=torch.long,
                device=device
            )
        )


    def hand_tile_emb(self, y, sep_embeddings, batch_size, device):
        cls_embeddings = self.special_token_emb(self.cls_token_id, batch_size, device)
        hand_tile_ids = y
        who_0_tile_token_type_embeddings = self.token_type_emb(
            14,
            self.who_0_tile_token_id,
            batch_size,
            device
        )
        who_1_tile_token_type_embeddings = self.token_type_emb(
            14,
            self.who_1_tile_token_id,
            batch_size,
            device
        )
        who_2_tile_token_type_embeddings = self.token_type_emb(
            14,
            self.who_2_tile_token_id,
            batch_size,
            device
        )
        who_3_tile_token_type_embeddings = self.token_type_emb(
            14,
            self.who_3_tile_token_id,
            batch_size,
            device
        )
        hand_tile_embeddings = self.tile_embeddings(hand_tile_ids)
        hand_tile_embeddings = torch.cat([
            cls_embeddings,
            hand_tile_embeddings[:, 0, :] + who_0_tile_token_type_embeddings,
            sep_embeddings,
            cls_embeddings,
            hand_tile_embeddings[:, 1, :] + who_1_tile_token_type_embeddings,
            sep_embeddings,
            cls_embeddings,
            hand_tile_embeddings[:, 2, :] + who_2_tile_token_type_embeddings,
            sep_embeddings,
            cls_embeddings,
            hand_tile_embeddings[:, 3, :] + who_3_tile_token_type_embeddings,
        ], dim=1)
        return hand_tile_embeddings


    def discards_tile_emb(self, x, sep_embeddings, batch_size, device):
        discards_tile_ids = x['discards']
        discards_tile_embeddings = self.tile_embeddings(discards_tile_ids)
        discards_len = discards_tile_embeddings.size()[2]
        position_embeddings = self.pos_emb(discards_len, batch_size, device)
        discards_who_0_token_type_embeddings = self.token_type_emb(
            discards_len,
            self.who_0_tile_token_id,
            batch_size,
            device
        )
        discards_who_1_token_type_embeddings = self.token_type_emb(
            discards_len,
            self.who_1_tile_token_id,
            batch_size,
            device
        )
        discards_who_2_token_type_embeddings = self.token_type_emb(
            discards_len,
            self.who_2_tile_token_id,
            batch_size,
            device
        )
        discards_who_3_token_type_embeddings = self.token_type_emb(
            discards_len,
            self.who_3_tile_token_id,
            batch_size,
            device
        )
        discards_tile_embeddings = torch.cat([
            discards_tile_embeddings[:, 0, :] + position_embeddings + discards_who_0_token_type_embeddings,
            sep_embeddings,
            discards_tile_embeddings[:, 1, :] + position_embeddings + discards_who_1_token_type_embeddings,
            sep_embeddings,
            discards_tile_embeddings[:, 2, :] + position_embeddings + discards_who_2_token_type_embeddings,
            sep_embeddings,
            discards_tile_embeddings[:, 3, :] + position_embeddings + discards_who_3_token_type_embeddings,
        ], dim=1)
        discards_tile_token_type_embeddings = self.token_type_emb(
            discards_tile_embeddings.size()[1],
            self.discards_tile_token_id,
            batch_size,
            device
        )
        return discards_tile_embeddings + discards_tile_token_type_embeddings


    def meld_emb(self, x):
        melds = x['melds']
        tile_embeddings = []
        type_embeddings = []

        for i in range(4):
            melds_ids = melds[i][:, 0, :]
            melds_type_ids = melds[i][:, 1, :]
            tile_embeddings.append(self.tile_embeddings(melds_ids))
            type_embeddings.append(self.token_type_embeddings(melds_type_ids))

        tile_embeddings_cat = torch.cat(tile_embeddings, dim=1)
        type_embeddings_cat = torch.cat(type_embeddings, dim=1)
        return tile_embeddings_cat + type_embeddings_cat



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
        h = self.dense0(hidden_states)
        h = self.act_fct(h)
        h = self.layer_norm(h)
        h = self.dense1(h)
        return h


class MahjongModelForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = 37
        self.embeddings = MahjongEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.head = BertHead(config, self.output_dim)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.mlm_probability = 0.15

    def forward(
        self,
        features,
        hand,
        attention_mask=None,
        token_type_id=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):

        x, token_type_ids, pos_ids = self.embeddings.data2x(hand, features, hand.device)
        # x, y = self.mask_tokens(
        #     x,
        #     mlm_probability=self.mlm_probability,
        #     special_token_id_list=self.embeddings.special_token_id_list,
        #     mask_token_id=self.embeddings.mask_token_id,
        #     device=hand.device
        # )
        x, y = self.mask_hand_tokens(
            x,
            mask_token_id=self.embeddings.mask_token_id
        )
        embedding_output = self.embeddings(x, token_type_ids, pos_ids)
        bert_outputs = self.bert_encoder(
            embedding_output
        )
        last_hidden_state = bert_outputs[0]
        logits = self.head(last_hidden_state)
        loss = self.loss_fct(logits.view(-1, self.output_dim), y.view(-1))
        batch_size = y.size()[0]
        accuracy = self.accuracy_fct(
            logits.view(-1, self.output_dim),
            y.view(-1)
        )
        return logits, loss, accuracy


    def accuracy_fct(self, logits, y):
        arg_sorted_logits = logits.argsort(descending=True)[:, :1]
        one_hot = F.one_hot(arg_sorted_logits, num_classes=self.output_dim)
        print(arg_sorted_logits[0])
        print(one_hot[0])
        return 1.0


    def mask_hand_tokens(self, inputs, mask_token_id):
        my_hand_idx = np.random.randint(0, 4)
        hand_len = 2 + 14
        batch_size = inputs.size()[0]
        device = inputs.device

        labels = inputs.clone()

        masked_indices = torch.zeros(labels.shape, device=device)
        masked_indices[:, : 4 * hand_len] = 1.0
        masked_indices[:, my_hand_idx * hand_len : (my_hand_idx + 1) * hand_len] = 0.0
        masked_indices = masked_indices.bool()

        labels[~masked_indices] = -100
        inputs[masked_indices] = mask_token_id

        return inputs, labels


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
