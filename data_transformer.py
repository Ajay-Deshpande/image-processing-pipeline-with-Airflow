import os
from glob import glob

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

from torchvision.transforms import v2, InterpolationMode
from torchvision.io import read_image, write_file, write_jpeg, write_png
from tqdm import tqdm
from airflow.decorators import task

@task(task_id = "image_processing")
def transform(data_path, transformed_images_path):
    image_folder = glob(f'{data_path}/*')

    transforms = v2.Compose([
        v2.RandomApply([
                        v2.ColorJitter(brightness = (.6, 1), contrast = (.65, .9), saturation = (0.3, 1), hue = (-.5, .5)),
                        v2.GaussianBlur(kernel_size = (9, 9), sigma = 1, )
                    ]),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomAffine(degrees = (-20, 20), scale = (.8, 1.2), translate = (0, .1)),
        v2.Resize((256, 256), InterpolationMode.BICUBIC)
    ])

    for folder_path in tqdm(image_folder):
        folder_class = folder_path.split('/')[-1]
        file_paths = glob(f'{folder_path}/*.jpg')
        aug_img_count = int(((300 - len(file_paths)) / len(file_paths)) + 1)
        for file_path in file_paths:
            file_name = file_path.split('.')[0].split('/')[-1]
            img = read_image(file_path)
            aug_ger = 0
            new_file_name = os.path.join(f'{transformed_images_path}', f'{folder_class}/{file_name}_aug_{aug_ger}.jpg')
            try:
                write_jpeg(v2.Resize((256, 256), InterpolationMode.BICUBIC)(img), new_file_name, quality = 100)
            except:
                os.makedirs('/'.join(new_file_name.split('/')[ : -1]))
                write_jpeg(v2.Resize((256, 256), InterpolationMode.BICUBIC)(img), new_file_name, quality = 100)
            for aug_ger in range(1, aug_img_count + 1):
                new_file_name = f'{folder_class}/{file_name}_aug_{aug_ger}.jpeg'
                new_file_name = os.path.join(f'{transformed_images_path}', new_file_name)
                try:
                    write_jpeg(transforms(img), new_file_name, quality = 100)
                except:
                    os.makedirs('/'.join(new_file_name.split('/')[ : -1]))
                    write_jpeg(transforms(img), new_file_name, quality = 100)
    return transformed_images_path

if __name__ == "__main__":
    transform("images", 'aug_imgs')