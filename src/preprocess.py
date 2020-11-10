
import pickle
import argparse
import os
from glob import glob
from tqdm import tqdm
import torch
from catalyst.utils import metrics
import catalyst
from mahjong.shanten import Shanten


parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str)
parser.add_argument('--input_path', type=str)
args = parser.parse_args()

pai_list = [
    "1m", "2m", "3m", "4m", "5m", "r5m", "6m", "7m", "8m", "9m", #萬子
    "1p", "2p", "3p", "4p", "5p", "r5p", "6p", "7p", "8p", "9p", #筒子
    "1s", "2s", "3s", "4s", "5s", "r5s", "6s", "7s", "8s", "9s", #索子
    "東", "南", "西", "北", "白", "發", "中"
]


def feature2xy(state, shanten_calculator):
    device = catalyst.utils.get_device()

    # action
    # Discard: 37, Reach:2 , Chow: 2, Pong: 2, Kong: 2
    x = {}

    positions = get_positions(state['action']['who'])

    # hand
    hand = state['hands'][state['action']['who']]
    hand = pais2ids(hand)
    x['hand'] = normalize_pai_list(hand, device)

    # discards : direction
    x['discards'] = normalize_discards(state['discards'], positions, device)

    # Shanten : direction
    x['shanten'], x['shanten_diff'] = calc_shantens(hand, device, shanten_calculator)

    if state['action']['type'] == 'discard':
        y = state['hands'][state['action']['who']].index(state['action']['tile'])


    elif state['action']['type'] == 'reach':
        y = state['action']['p']

    elif state['action']['type'] == 'chow':
        y = state['action']['p']

    elif state['action']['type'] == 'pong':
        y = state['action']['p']

    elif state['action']['type'] == 'kong':
        y = state['action']['p']

    # melds : direction
    x['melds'] = [
        normalize_melds([
            m for m in state['melds'] if m['who'] == i
        ], device) for i in positions
    ]

    # action_meld_tiles
    if state['action']['type'] in ['chow', 'pong', 'kong']:
        x['action_meld_tiles'] = normalize_action_meld_tiles(state['action']['meld_state']['meld_tiles'], device)
    else:
        x['action_meld_tiles'] = torch.zeros(4, dtype=torch.long, device=device)

    # menzen : direction
    x['menzen'] = torch.tensor(
        [state['menzen'][i] for i in positions],
        dtype=torch.long, device=device
    )

    # reach_state : direction
    x['reach_state'] = torch.tensor(
        [state['reach_state'][i] for i in positions],
        dtype=torch.long, device=device
    )

    # n_reach
    x['n_reach'] = torch.tensor([min([state['n_reach'], 2])], dtype=torch.long, device=device)

    # reach_ippatsu : direction
    x['reach_ippatsu'] = torch.tensor(
        [state['reach_ippatsu'][i] for i in positions],
        dtype=torch.long, device=device
    )

    # doras
    x['doras'] = normalize_doras(state['doras'], device)

    # dans : direction
    x['dans'] = torch.tensor(
        [state['dans'][i] for i in positions],
        dtype=torch.long, device=device
    )

    # rates : direction
    x['rates'] = normalize_rates(state['rates'], positions, device=device)

    # oya : direction
    x['oya'] = torch.tensor([(state['oya'] - state['action']['who'] + 4) % 4], dtype=torch.long, device=device)

    # scores : direction
    x['scores'] = normalize_scores(state['scores'], positions, device)

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
    x['who'] = torch.tensor([state['action']['who']], dtype=torch.long, device=device)

    x['sum_discards'] = torch.tensor(
        [calc_sum_discards(state['discards'])],
        dtype=torch.long,
        device=device
    )

    return { 'x': x, 'y': y }


def calc_sum_discards(discards):
    #  0から11 : 0
    # 12から23 : 1
    # 24から35 : 2
    # 36から47 : 3
    # 48から59 : 4
    # 60以上   : 5
    sum_discards = sum([len(d) for d in discards])

    if sum_discards >= 60:
        return 5

    return sum_discards // (4 * 3)

def normalize_score(score):
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

def normalize_scores(scores, positions, device):
    return torch.tensor(
        [normalize_score(scores[i]) for i in positions],
        dtype=torch.long, device=device
    )

def normalize_rate(rate):
    rate = int(rate) // 100 - 14

    if rate < 0:
        return 0

    if rate > 9:
        return 9

    return rate

def normalize_rates(rates, positions, device):
    # id : rate range
    # 0 : 1499以下
    # 1 : 1500から1600
    # …
    # 9 : 2300以上

    return torch.tensor(
        [normalize_rate(rates[i]) for i in positions],
        dtype=torch.long, device=device
    )


def normalize_doras(doras, device, max_len=5):
    doras_tensor = torch.zeros(max_len, dtype=torch.long, device=device)
    l = len(doras)
    doras = pais2ids(doras)
    doras_tensor[:l] = torch.tensor(doras, dtype=torch.long, device=device)
    return doras_tensor

def normalize_action_meld_tiles(tiles, device, max_len=4):
    tiles_tensor = torch.zeros(max_len, dtype=torch.long, device=device)
    l = len(tiles)
    tiles = pais2ids(tiles)
    tiles_tensor[:l] = torch.tensor(tiles, dtype=torch.long, device=device)

    return tiles_tensor

def normalize_melds(melds, device, max_len=20):
    meld_tensor_list = [meld2tensor(meld, device) for meld in melds]
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

def meld2tensor(meld, device):
    if meld['meld_type'] == 2:
        # Add Kan
        tiles = [pais2ids(meld['meld_tiles'])[0]]
    else:
        tiles = pais2ids(meld['meld_tiles'])

    l = len(tiles)
    tile_ids = torch.tensor(tiles, dtype=torch.long, device=device)
    token_type = torch.full((l, ), fill_value=meld['meld_type'], dtype=torch.long, device=device)
    return torch.tensor([*tile_ids, *token_type]).reshape([2, -1])


def normalize_discards(discards, positions, device):
    max_n_discards = 25
    l = len(discards)
    res = torch.zeros((4, max_n_discards), dtype=torch.long, device=device)

    for i, pos in enumerate(positions):
        res[i, :len(discards[pos])] = torch.tensor(
            pais2ids(discards[pos]),
            dtype=torch.long,
            device=device
        )
    return res


def calc_shantens(hand, device, shanten_calculator):
    shantens = []
    hand_34_count = to_34_count(hand)
    hand_34 = to_34_array(hand)
    base_shanten, _ = calc_shanten(hand_34_count, shanten_calculator)

    l = len(shantens)
    x = torch.full((14, ), fill_value=-1, dtype=torch.long, device=device)

    base_shanten = min([base_shanten, 6]) + 1
    base_shanten = torch.tensor([base_shanten], dtype=torch.long, device=device)

    return base_shanten, x


def to_34_array(hand):
    return [to_34(t) for t in hand]

def to_34_count(hand):
    res = [0] * 34
    for tile in hand:
        res[to_34(tile)] += 1
    return res

def to_34(tile):
    if tile <= 5:
        return tile - 1
    if tile <= 15:
        return tile - 2
    if tile <= 25:
        return tile - 3
    return tile - 4


def calc_shanten(tiles_34, shanten_calculator, open_sets_34=None):
    shanten_with_chiitoitsu = shanten_calculator.calculate_shanten(tiles_34,
                                                                    open_sets_34,
                                                                    chiitoitsu=True)
    shanten_without_chiitoitsu = shanten_calculator.calculate_shanten(tiles_34,
                                                                        open_sets_34,
                                                                        chiitoitsu=False)

    return min([shanten_with_chiitoitsu, shanten_without_chiitoitsu]), shanten_with_chiitoitsu <= shanten_without_chiitoitsu


def normalize_pai_list(pai_list, device, n=14):
    l = len(pai_list)
    x = torch.zeros(n, dtype=torch.long, device=device)
    x[:l] = torch.tensor(pai_list, dtype=torch.long, device=device)
    return x

def pais2ids(pai_list):
    return [pai2id(pai) for pai in pai_list]

def pai2id(pai):
    pai_list = [
        "1m", "2m", "3m", "4m", "5m", "r5m", "6m", "7m", "8m", "9m", #萬子
        "1p", "2p", "3p", "4p", "5p", "r5p", "6p", "7p", "8p", "9p", #筒子
        "1s", "2s", "3s", "4s", "5s", "r5s", "6s", "7s", "8s", "9s", #索子
        "東", "南", "西", "北", "白", "發", "中"
    ]
    assert(pai in pai_list)
    return pai_list.index(pai) + 1

def get_positions(who, n_players=4):
    return [(i + who) % n_players for i in range(n_players)]


if __name__ == '__main__':
    path_list = glob(args.input_path)
    outputs_count = 0
    shanten_calculator = Shanten()

    print(f'Input Size : {len(path_list)}')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for i, path in tqdm(enumerate(path_list)):
        with open(path, 'rb') as f:
            features = pickle.load(f)
        path_without_ext = os.path.splitext(path)[0]
        dump_xy = [feature2xy(feature, shanten_calculator) for feature in features]

        with open(f'{args.output_path}/{i:05}.pt', 'wb') as f:
            pickle.dump(dump_xy, f)

        outputs_count += 1

    print(f'final outputs count:{outputs_count}')
