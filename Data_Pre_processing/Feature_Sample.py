import random


def Feature_Sample(images, window_size):
    max_num = images.shape[-1] - window_size
    h, w = random.sample(range(max_num), 2)
    return images[:, :, h:h + window_size, w:w + window_size]

