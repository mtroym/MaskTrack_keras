import numpy as np
from matplotlib import pyplot as plt

from model import Deeplabv3

if __name__ == '__main__':
    deeplab_model = Deeplabv3(weights='pascal_voc', backbone='xception', OS=8)
    img = plt.imread("./test/test-0.PNG")
    img = img[..., :3]
    # w, h, _ = img.shape
    # ratio = 512. / np.max([w, h])
    # resized = cv2.resize(img, (int(ratio * h), int(ratio * w)))
    # resized = resized / 127.5 - 1.
    pad_x = int(512 - img.shape[0])
    pad_y = int(512 - img.shape[1])
    resized2 = np.pad(img, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
    score = deeplab_model.predict(np.expand_dims(resized2, 0))
    preds = np.argmax(score.squeeze(), -1)[:-pad_x, :-pad_y]
    plt.imshow(img)
    plt.show()
    print(np.unique(preds))
    mask = np.zeros_like(img)
    mask[preds == 15] = 1
    alpha = 0.5
    img_ = (1 - mask) * img + mask * (np.array([1, 0, 0]) * 0.5 + 0.5 * img)
    plt.imshow(img_)
    plt.show()
