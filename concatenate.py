import h5py
import numpy as np
import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features1_path', type=str)
    parser.add_argument('--features2_path', type=str)
    parser.add_argument('--save_path', type=str)

if __name__ == '__main__':
    args = get_arg()

    print('____Start____')
    print('____Concatenating____')
    v3c1 = h5py.File(args.features1_path)
    v3c1_ids = v3c1.get('ids')
    v3c1_features = v3c1.get('features')

    v3c1_ids = np.array(v3c1_ids)
    v3c1_features = np.array(v3c1_features)

    
    v3c2 = h5py.File(args.features2_path)
    v3c2_ids = v3c2.get('ids')
    v3c2_features = v3c2.get('features')

    v3c2_ids = np.array(v3c2_ids)
    v3c2_features = np.array(v3c2_features)

    ids = np.concatenate((v3c1_ids, v3c2_ids))
    features = np.concatenate((v3c1_features, v3c2_features))

    features = features.tolist()
    ids = ids.tolist()

    save_path = args.save_path
    data = h5py.File(save_path, 'w')
    data.create_dataset('ids', data=ids)
    data.create_dataset('features', data=features)
    data.close()
    print('___Finished____')

