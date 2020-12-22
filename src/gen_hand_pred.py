import argparse
import pickle
import torch
import json
import os
import math
from glob import glob
from torch.utils.data import DataLoader
from tqdm import tqdm

from hand_pred_model import get_model
from beam_search import beam_search
from hand_pred_data import PaifuDataset


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='./')
parser.add_argument('--input_path', type=str, default='./')
parser.add_argument('--output_path', type=str, default='./')
parser.add_argument('--n_outputs', type=int, default=100)
parser.add_argument('--n_max', type=int, default=100)
parser.add_argument('--beam_width', type=int, default=4)
args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)


def tensor_to_serializable(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy().tolist()
    elif isinstance(t, dict):
        return all_dict_elm_tensor_to_serializable(t)
    elif isinstance(t, list):
        return all_list_elm_tensor_to_serializable(t)
    return t

def all_dict_elm_tensor_to_serializable(d):
    res = {}
    for k in d.keys():
        res[k] = tensor_to_serializable(d[k])
    return res

def all_list_elm_tensor_to_serializable(d):
    res = []
    for k in d:
        res.append(tensor_to_serializable(k))
    return res


if __name__ == '__main__':
    model = get_model()
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)

    n_paths = math.ceil(args.n_outputs / args.n_max)
    path_list = glob(args.input_path)
    path_list = path_list[:n_paths]
    dataset = PaifuDataset(path_list, n_max=args.n_max)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f'len(dataset):{len(dataset)}')
    print(f'n_paths:{n_paths}')

    for x, _ in tqdm(loader):
        n_hands_list = x['n_hands_list']

        search_result = beam_search(n_hands_list, model, x, beam_width=args.beam_width)

        save_dict = {
            'x': all_dict_elm_tensor_to_serializable(x),
            'search_result': all_dict_elm_tensor_to_serializable(search_result)
        }
        with open(f'{args.output_path}/{x["paifu_id"][0]}.json', 'w') as f:
            json.dump(save_dict, f)
