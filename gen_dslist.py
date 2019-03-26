import os
from PIL import Image
import numpy as np
if __name__ == '__main__':
    dataset_root = '/Users/tony/data/DAVIS/'
    for split in ['train', 'val']:
        split_list_folders = []
        paths_all = []
        for year in [2016, 2017]:
            image_sets = os.path.join(dataset_root, 'ImageSets', str(year), split + '.txt')
            with open(image_sets, 'r') as f:
                split_list = f.read().split()
            split_list_folders += split_list
        split_list_folders = sorted(list(set(split_list_folders)))
        for folder in split_list_folders:
            image_path = os.path.join(dataset_root, "JpegImages", "480p", folder)
            annot_path = os.path.join(dataset_root, "Annotations", "480p", folder)
            paths = []
            for p in sorted(os.listdir(image_path)):
                if not p.endswith('.jpg'):
                    continue
                path_str = os.path.join(image_path, p) + " " + os.path.join(annot_path, p.replace('.jpg', '.png'))
                groud_truth = np.unique(np.array(Image.open(path_str.split()[-1]), dtype=np.uint8))
                for pix in groud_truth:
                    if pix == 0:
                        continue
                    paths.append(path_str + ' ' + str(pix))
            paths_all += paths
        print(split, len(paths_all))
        with open('../MT_split/'+split + '.txt', 'w') as f:
            f.write('\n'.join(paths_all))
            f.close()
