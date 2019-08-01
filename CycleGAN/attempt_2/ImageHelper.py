import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from glob import glob
from imageio import imread
from skimage.transform import resize

class ImageHelper:
    
    @staticmethod
    def load_images(path, dataset_type = 'train', batch_size = 1, image_resolution = (128, 128)):
        path_A = np.array(glob(path + dataset_type + 'A/*'))
        path_B = np.array(glob(path + dataset_type + 'B/*'))
        path_A_samples = list(path_A[np.random.randint(0, len(path_A), batch_size)])
        path_B_samples = list(path_B[np.random.randint(0, len(path_B), batch_size)])
        images_A, images_B = [], []
        for image_path_A, image_path_B in zip(path_A_samples, path_B_samples):
            image_A = resize(imread(image_path_A).astype(np.float), image_resolution)
            image_B = resize(imread(image_path_B).astype(np.float), image_resolution)
            if dataset_type == 'train' and np.random.random() > 0.5:
                image_A = np.fliplr(image_A)
                image_B = np.fliplr(image_B)
            images_A.append(image_A)
            images_B.append(image_B)
        images_A = np.array(images_A) / 127.5 - 1.0
        images_B = np.array(images_B) / 127.5 - 1.0
        return images_A, images_B
    
    @staticmethod
    def load_batch(path, dataset_type = 'train', batch_size = 1, image_resolution = (128, 128)):
        path_A = np.array(glob(path + dataset_type + 'A/*'))
        path_B = np.array(glob(path + dataset_type + 'B/*'))
        path_A_samples = list(path_A[np.random.randint(0, len(path_A), batch_size)])
        path_B_samples = list(path_B[np.random.randint(0, len(path_B), batch_size)])
        images_A, images_B = [], []
        for image_path_A, image_path_B in zip(path_A_samples, path_B_samples):
            image_A = resize(imread(image_path_A).astype(np.float), image_resolution)
            image_B = resize(imread(image_path_B).astype(np.float), image_resolution)
            if dataset_type == 'train' and np.random.random() > 0.5:
                image_A = np.fliplr(image_A)
                image_B = np.fliplr(image_B)
            images_A.append(image_A)
            images_B.append(image_B)
        images_A = np.array(images_A) / 127.5 - 1.0
        images_B = np.array(images_B) / 127.5 - 1.0
        images_A = images_A.reshpae(batch_size, image_resolution[0], image_resolution[1], 3)
        yield images_A, images_B
    
    def visualize_batch(self, rows, cols, figsize, images_A, images_B, A_title, B_title):
        fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = figsize)
        plt.setp(axes.flat, xticks = [], yticks = [])
        c = 1
        for i, ax in enumerate(axes.flat):
            if i % 2 == 0:
                ax.imshow(images_A[c])
                ax.set_xlabel(A_title + '_' + str(c))
            else:
                ax.imshow(images_B[c])
                ax.set_xlabel(B_title + '_' + str(c))
                c += 1
plt.show()