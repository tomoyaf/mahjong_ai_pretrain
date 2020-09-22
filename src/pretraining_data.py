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
        x = paifu[state_idx]
        # who_x = np.random.randint(0, len(x['hands']))

        device = catalyst.utils.get_device()
        # x['my_hand'] = self.pais2ids(x['hands'][who_x])
        # x['my_hand'] = self.normalize_pai_list(x['my_hand'], device)

        # x['known_other_player_hands'] = torch.zeros((3, 14), dtype=torch.long)
        # y = torch.zeros((3, 14), device=device, dtype=torch.long)

        y = torch.zeros((4, 14), device=device, dtype=torch.long)

        for i, hand in enumerate(x['hands']):
            hand = self.pais2ids(hand)
            y[i, :] = self.normalize_pai_list(hand, device)

        # for i, hand in enumerate(x['hands']):
        #     if i == who_x:
        #         continue

            # j = (4 + i - who_x) % 4 - 1

            # l = len(hand)
            # p = np.random.uniform(0.0, 1.0)
            # p = 0.1
            # mask = np.array([True] + [True] * int((l - 1) * p) + [False] * int((l - 1) * (1.0 - p)))
            # np.random.shuffle(mask)

            # x_known = self.pais2ids(
            #     [h for h, m in zip(hand, mask) if not m]
            # )
            # x['known_other_player_hands'][j, :len(x_known)] = torch.tensor(x_known, device=device)

            # y_arr = self.pais2ids(
            #     [h for h, m in zip(hand, mask) if m]
            # )

            # if i < who_x:
            #     y[j, :len(y_arr)] = torch.tensor(y_arr, device=device)
            # elif i > who_x:
            #     y[j-1, :len(y_arr)] = torch.tensor(y_arr, device=device)

        x['discards'] = self.normalize_discards(x['discards'], device)
        x['melds'] = [
            self.normalize_melds([
                m for m in x['melds'] if m['who'] == i
            ], device) for i in range(4)
        ]
        x['doras'] = self.normalize_doras(x['doras'], device)

        return x, y

    def normalize_doras(self, doras, device, max_len=5):
        doras_tensor = torch.zeros(max_len, dtype=torch.long, device=device)
        l = len(doras)
        doras = self.pais2ids(doras)
        doras_tensor[:l] = torch.tensor(doras, dtype=torch.long, device=device)
        return doras_tensor

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
        return [self.pai_list.index(pai) for pai in pai_list]
