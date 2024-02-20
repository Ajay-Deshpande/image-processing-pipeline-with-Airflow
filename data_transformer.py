import os
from glob import glob

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

from torchvision.transforms import v2, InterpolationMode
from torchvision.io import read_image, write_file, write_jpeg, write_png
from tqdm import tqdm

def main():
    data_path = glob('images/*')

    transforms = v2.Compose([
        v2.RandomApply([
                        v2.ColorJitter(brightness = (.6, 1), contrast = (.65, .9), saturation = (0.3, 1), hue = (-.5, .5)),
                        v2.GaussianBlur(kernel_size = (9, 9), sigma = 1, )
                    ]),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomAffine(degrees = (-20, 20), scale = (.8, 1.2), translate = (0, .1)),
        v2.Resize((256, 256), InterpolationMode.BICUBIC)
    ])

    for folder_path in tqdm(data_path):
        folder_class = folder_path.split('\\')[-1]
        file_paths = glob(f'images/{folder_class}/*.jpg')
        aug_img_count = int(((300 - len(file_paths)) / len(file_paths)) + 1)
        for file_path in file_paths:
            file_name = file_path.split('.')[0].split('\\')[-1]
            img = read_image(file_path)
            for aug_ger in range(aug_img_count):
                new_file_name = f'augmented_images/{folder_class}/{file_name}_aug_{aug_ger}.jpeg'
                try:
                    write_jpeg(transforms(img), new_file_name, quality = 100)
                except:
                    os.makedirs('/'.join(new_file_name.split('/')[ : -1]))
                    write_jpeg(transforms(img), new_file_name, quality = 100)
            new_file_name = f'augmented_images/{folder_class}/{file_name}.jpeg'
            try:
                write_jpeg(v2.Resize((256, 256), InterpolationMode.BICUBIC)(img), new_file_name, quality = 100)
            except:
                os.makedirs('/'.join(new_file_name.split('/')[ : -1]))
                write_jpeg(v2.Resize((256, 256), InterpolationMode.BICUBIC)(img), new_file_name, quality = 100)

if __name__ == "__main__":
    main()