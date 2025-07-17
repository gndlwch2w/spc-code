import random
import numpy as np
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from PIL import Image, ImageFilter

def random_rot_flip(image, label=None):
    """Randomly rotate and flip the image and label."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label):
    """Randomly rotate the image and label by a random angle."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def color_jitter(image):
    """Apply color jitter to the image."""
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def blur(image, p=0.5):
    """Apply Gaussian blur to the image with a probability p."""
    if random.random() < p:
        max = np.max(image)
        min = np.min(image)
        sigma = np.random.uniform(0.1, 2.0)
        image = Image.fromarray(((image - min) / (max - min) * 255).astype('uint8'))
        image = np.array(image.filter(ImageFilter.GaussianBlur(radius=sigma)))
        image = min + image * (max - min) / 255
    return image

class RandomGenerator:
    """Randomly generate an image and label with specified output size."""
    
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        """
        :param sample: {"image": [h, w], "label": [h, w]}
        :return: {"image": [h, w], "label": [h, w]}
        """
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image = self.resize(image)
        label = self.resize(label)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        return {"image": image, "label": label}

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

class WeakStrongAugment:
    """Apply weak and strong augmentations to the image and label."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # weak augmentation is rotation / flip
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # strong augmentation is color jitter
        image_strong = image
        label_strong = label

        image_strong = color_jitter(image_strong).type("torch.FloatTensor")
        image_strong = image_strong[0].numpy()
        image_strong = blur(image_strong, p=0.5)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        label_strong = torch.from_numpy(label_strong.astype(np.uint8))

        sample = {
            "image_w": image,
            "image_s": image_strong,
            "label_w": label,
            "label_s": label_strong
        }
        return sample
