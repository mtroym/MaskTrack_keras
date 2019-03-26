import random

import cv2
import keras
import numpy as np
from PIL import Image
from albumentations import (
    Compose,
    ElasticTransform,
    IAAPiecewiseAffine
)
from keras.utils import to_categorical
from matplotlib import pyplot as plt

debug = 1


class DataGenerator(keras.utils.Sequence):
    def __init__(self, file_path, batch_size=32, n_channels=3, n_classes=2, shuffle=True, input_shape=(854, 480)):
        self.batch_size = batch_size
        f = open(file_path, 'r')
        self.list_paths = f.readlines()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.input_shape = input_shape
        self.indexes = np.arange(len(self.list_paths))
        self.on_epoch_end()

    @staticmethod
    def mask_transformation(mask):
        """
        To simulate the deformation noise. affine transform + non-rigid deformations(TPS) + dilation
         This is from paper: `during offline training we generate input masks by deforming the
                             annotated masks via affine transformation as well as non-rigid
                             deformations via thin-plate splines [4], followed by a coarsening
                             step (dilation morphological operation) to remove details of the object contour. `

        :param mask: np.array of shape (h, w)
        :return: np.array of shape (h, w) after deformation noise.
        """
        aug = Compose([
            # add another mask transformation ?
            IAAPiecewiseAffine(p=0.4),  # affine + non-rigid deformation
            ElasticTransform(p=0.6, alpha=50, sigma=50, alpha_affine=20,
                             border_mode=cv2.BORDER_CONSTANT, always_apply=True),
        ])
        augmented = aug(image=mask)
        mask = augmented['image']
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=random.randrange(1, 10))
        return mask

    def __len__(self):
        return int(np.floor(len(self.list_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_path_batch = [self.list_paths[k] for k in indexes]
        return self.__data_generation(list_path_batch)  # load from file.

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths):
        image_mask_slice = []
        masks = []
        # Generate data
        for i, info in enumerate(list_paths):
            (path_img, path_gt, instance_id) = info.split()
            instance_id = int(instance_id)
            labels = Image.open(path_gt)
            image = Image.open(path_img)

            labels.load()
            image.load()

            labels = labels.resize(self.input_shape)  # contains multiple instance.
            image = image.resize(self.input_shape)

            labels = np.array(labels, dtype=np.uint8)  # contains multiple instance.
            image = np.array(image, dtype=np.uint8)

            ground_truth = np.zeros_like(labels)
            ground_truth[labels == instance_id] = 1  # single instance.

            # Augmentation for mask input; single instance.
            mask = self.mask_transformation(ground_truth)
            if debug:
                mask_ = np.stack((mask, mask, mask), -1)
                img_ = (1 - mask_.astype(np.float)) * image.astype(np.float) + mask_.astype(np.float) * (
                        np.array([255.0, 0, 0]) * 0.3 + 0.7 * image.astype(np.float))
                plt.imshow(img_.astype(np.int))
                plt.show()
            mask = mask * 400 - 200  # scale 0 ~ 1 to -200 ~ 200
            image_mask = np.concatenate((image, mask[..., np.newaxis]), 2)

            # adjust the dimension for the binary crossentropy loss.
            ground_truth = to_categorical(ground_truth, 2)

            # accumulation data.
            image_mask_slice.append(image_mask)
            masks.append(ground_truth)

        return np.stack(image_mask_slice, 0), np.stack(masks, 0)


if __name__ == '__main__':
    random.seed(10)
    # dataloader = Dataset(train_list='train.txt', test_list='val.txt', database_root='.', store_memory=False,
    #                      data_aug=False)
    # print(dataloader.next_batch(20, 'train'))
    print(random.randrange(1, 10))
    datagen = DataGenerator('../MT_split/train.txt', batch_size=2)
    print(len(datagen))
    x, y = datagen[0]
    print(x.shape, y.shape)
