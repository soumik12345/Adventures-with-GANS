from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

generator = load_model('./models/generator.h5')
generator.summary()

def plot_images(nrows, ncols, figsize, generator):
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    plt.setp(axes.flat, xticks = [], yticks = [])
    noise = np.random.normal(0, 1, (nrows * ncols, 100))
    generated_images = generator.predict(noise).reshape(nrows * ncols, 28, 28)
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i], cmap = 'gray')
    plt.show()

plot_images(2, 8, (16, 6), generator)