import torch
import torch.nn.functional as F
import numpy as np
import pickle
import catalyst


class PaifuDataset(torch.utils.data.Dataset):
    pai_list = [
        "1m", "2m", "3m", "4m", "5m", "r5m", "6m", "7m", "8m", "9m", #萬子
        "1p", "2p", "3p", "4p", "5p", "r5p", "6p", "7p", "8p", "9p", #筒子
        "1s", "2s", "3s", "4s", "5s", "r5s", "6s", "7s", "8s", "9s", #索子
        "東", "南", "西", "北", "白", "發", "中"
    ]

    def __init__(self, paifu_path_list):
        self.paifu_path_list = paifu_path_list
        self.data_size = len(paifu_path_list)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        paifu = None

        with open(self.paifu_path_list[idx], 'rb') as f:
            paifu = pickle.load(f)

        l = len(paifu)
        state_idx = np.random.randint(0, l)
        x, y = self.paifu_to_xy(paifu, state_idx)
        return x, y

    def paifu_to_xy(self, paifu, state_idx):
        device = catalyst.utils.get_device()
        state = paifu[state_idx]

        # action
        # Discard: 37, Reach:2 , Chow: 2, Pong: 2, Kong: 2
        y = torch.full((5, ), -100, dtype=torch.long, device=device)
        x = {}

        if state['action']['type'] == 'discard':
            discarded_idx = self.pai_list.index(state['action']['tile'])
            y[0] = discarded_idx

        elif state['action']['type'] == 'reach':
            y[1] = state['action']['p']

        elif state['action']['type'] == 'chow':
            y[2] = state['action']['p']

        elif state['action']['type'] == 'pong':
            y[3] = state['action']['p']

        elif state['action']['type'] == 'kong':
            y[4] = state['action']['p']


        # hand
        hand = state['hands'][state['action']['who']]
        hand = self.pais2ids(hand)
        x['hand'] = self.normalize_pai_list(hand, device)
        x['discards'] = self.normalize_discards(state['discards'], device)

        # melds
        x['melds'] = [
            self.normalize_melds([
                m for m in state['melds'] if m['who'] == i
            ], device) for i in range(4)
        ]

        # action_meld_tiles
        if state['action']['type'] in ['chow', 'pong', 'kong']:
            x['action_meld_tiles'] = self.normalize_action_meld_tiles(state['action']['meld_state']['meld_tiles'], device)
        else:
            x['action_meld_tiles'] = torch.zeros(4, dtype=torch.long, device=device)

        # menzen
        x['menzen'] = torch.tensor(state['menzen'], dtype=torch.long, device=device)

        # reach_state
        x['reach_state'] = torch.tensor(state['reach_state'], dtype=torch.long, device=device)

        # n_reach
        x['n_reach'] = torch.tensor([min([state['n_reach'], 2])], dtype=torch.long, device=device)

        # reach_ippatsu
        x['reach_ippatsu'] = torch.tensor(state['reach_ippatsu'], dtype=torch.long, device=device)

        # doras
        x['doras'] = self.normalize_doras(state['doras'], device)

        # dans
        x['dans'] = torch.tensor(state['dans'], dtype=torch.long, device=device)

        # rates
        x['rates'] = self.normalize_rates(state['rates'], device=device)

        # oya
        x['oya'] = torch.tensor([state['oya']], dtype=torch.long, device=device)

        # scores
        x['scores'] = self.normalize_scores(state['scores'], device)

        # n_honba
        x['n_honba'] = torch.tensor([min([state['n_honba'], 3])], dtype=torch.long, device=device)

        # n_round
        x['n_round'] = torch.tensor([state['n_round']], dtype=torch.long, device=device)

        # sanma_or_yonma
        x['sanma_or_yonma'] = torch.tensor([state['sanma_or_yonma']], dtype=torch.long, device=device)

        # han_or_ton
        x['han_or_ton'] = torch.tensor([state['han_or_ton']], dtype=torch.long, device=device)

        # aka_ari
        x['aka_ari'] = torch.tensor([state['aka_ari']], dtype=torch.long, device=device)

        # kui_ari
        x['kui_ari'] = torch.tensor([state['kui_ari']], dtype=torch.long, device=device)

        # who
        x['who'] = state['action']['who']

        return x, y
        # return torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])

    def normalize_score(self, score):
        # 0 : 4000以下（満貫は2000-4000だから）
        # 1 : 4001から8000
        # …
        # 11 : 44000から48000
        # 12 : 48001以上

        score = (score - 1) // 4000

        if score < 0:
            return 0

        if score > 12:
            return 12

        return score

    def normalize_scores(self, scores, device):
        return torch.tensor([self.normalize_score(score) for score in scores], dtype=torch.long, device=device)

    def normalize_rate(self, rate):
        rate = int(rate) // 100 - 14

        if rate < 0:
            return 0

        if rate > 9:
            return 9

        return rate

    def normalize_rates(self, rates, device):
        # id : rate range
        # 0 : 1499以下
        # 1 : 1500から1600
        # …
        # 9 : 2300以上

        return torch.tensor([self.normalize_rate(rate) for rate in rates], dtype=torch.long, device=device)


    def normalize_doras(self, doras, device, max_len=5):
        doras_tensor = torch.zeros(max_len, dtype=torch.long, device=device)
        l = len(doras)
        doras = self.pais2ids(doras)
        doras_tensor[:l] = torch.tensor(doras, dtype=torch.long, device=device)
        return doras_tensor

    def normalize_action_meld_tiles(self, tiles, device, max_len=4):
        tiles_tensor = torch.zeros(max_len, dtype=torch.long, device=device)
        l = len(tiles)
        tiles = self.pais2ids(tiles)
        tiles_tensor[:l] = torch.tensor(tiles, dtype=torch.long, device=device)

        return tiles_tensor

    def normalize_melds(self, melds, device, max_len=20):
        meld_tensor_list = [self.meld2tensor(meld, device) for meld in melds]
        meld_tensor = torch.zeros((2, max_len), dtype=torch.long, device=device)

        if len(meld_tensor_list) < 1:
            return meld_tensor

        meld_tensor_cat = torch.cat(meld_tensor_list, dim=1)
        l = meld_tensor_cat.size()[1]
        if l > max_len:
            print('Invalid meld data.')
            print(meld_tensor_cat)
        meld_tensor[:, :l] = meld_tensor_cat
        return meld_tensor

    def meld2tensor(self, meld, device):
        if meld['meld_type'] == 2:
            # Add Kan
            tiles = [self.pais2ids(meld['meld_tiles'])[0]]
        else:
            tiles = self.pais2ids(meld['meld_tiles'])

        l = len(tiles)
        tile_ids = torch.tensor(tiles, dtype=torch.long, device=device)
        token_type = torch.full((l, ), fill_value=meld['meld_type'], dtype=torch.long, device=device)
        return torch.tensor([*tile_ids, *token_type]).reshape([2, -1])


    def normalize_discards(self, discards, device):
        max_n_discards = 25
        l = len(discards)
        res = torch.zeros((4, max_n_discards), dtype=torch.long, device=device)

        for i in range(l):
            res[i, :len(discards[i])] = torch.tensor(
                self.pais2ids(discards[i]),
                dtype=torch.long,
                device=device
            )
        return res


    def normalize_pai_list(self, pai_list, device, n=14):
        l = len(pai_list)
        x = torch.zeros(n, dtype=torch.long, device=device)
        x[:l] = torch.tensor(pai_list, dtype=torch.long, device=device)
        return x

    def pais2ids(self, pai_list):
        return [self.pai2id(pai, pai_list) for pai in pai_list]

    def pai2id(self, pai, pai_list=[]):
        if pai not in self.pai_list:
            print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
            print(pai)
            print(pai_list)
            [][-1]

        return self.pai_list.index(pai) + 1
