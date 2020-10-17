
import pickle
import argparse
import os
from glob import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str)
parser.add_argument('--input_path', type=str)
args = parser.parse_args()


if __name__ == '__main__':
    path_list = glob(args.input_path)[:10]

    for i, path in tqdm(enumerate(path_list)):
        with open(path, 'rb') as f:
            paifu = pickle.load(f)
        path_without_ext = os.path.splitext(path)[0]
        for j, d in enumerate(paifu):
            with open(f'{args.output_path}/{i:08}_{j:08}.pickle', 'wb') as f:
                pickle.dump(d, f)


    with open(f'{args.output_path}/00000000_00000000.pickle', 'rb') as f:
        res = pickle.load(f)
        print(res)