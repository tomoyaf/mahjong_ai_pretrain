
import pickle
import argparse
import os
from glob import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--n_outputs', type=int)
parser.add_argument('--n_max', type=int)
args = parser.parse_args()


if __name__ == '__main__':
    path_list = glob(args.input_path)
    outputs_count = 0
    stacked = []

    print(f'Input Size : {len(path_list)}')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for i, path in tqdm(enumerate(path_list)):
        with open(path, 'rb') as f:
            paifu = pickle.load(f)
        path_without_ext = os.path.splitext(path)[0]
        for j, d in enumerate(paifu):
            stacked.append(d)

            if len(stacked) >= args.n_max:
                with open(f'{args.output_path}/{args.n_max}_{outputs_count}.pickle', 'wb') as f:
                    pickle.dump(stacked, f)

                outputs_count += 1
                stacked = []

            if outputs_count >= args.n_outputs:
                break
        if outputs_count >= args.n_outputs:
            break
