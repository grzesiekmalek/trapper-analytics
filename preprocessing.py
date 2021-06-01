import os
import pathlib

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import random

labels = ['Wild Boar',
          'Red Deer',
          'Eurasian Elk',
          'Eurasian Red Squirrel',
          'Red Fox',
          'Roe Deer',
          'European Pine Marten',
          'Raccoon Dog',
          'European Bison',
          'Domestic Cat',
          'Wolf',
          'European Badger']


class Preprocessing:
    BATCH_SIZE=32
    def __init__(self):
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    def image_generator(self):
        wanted_imgs_cnt_per_class = 4000
        data_dir = pathlib.Path('input')

        for label in labels:
            all_imgs_per_class = os.listdir(data_dir / label)
            print(label)
            imgs_in_class = len(os.listdir(data_dir / label))
            print(f'Number of imgs in class {label} is {imgs_in_class}.')
            images_to_generate = wanted_imgs_cnt_per_class - imgs_in_class
            original_imgs = [x for x in all_imgs_per_class if not x.startswith('augmented')]

            for t in range(images_to_generate):
                path_to_original_image = data_dir / label / random.choice(original_imgs)
                print(path_to_original_image)
                with open(path_to_original_image) as img:
                    img = load_img(data_dir / label / random.choice(original_imgs))
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)
                    self.datagen.flow(x, batch_size=1, save_to_dir=data_dir / label, save_prefix='augmented',
                                 save_format='jpg').__next__()

    def image_flow_generator(self):
        wanted_imgs_cnt_per_class = 2000
        data_dir = pathlib.Path('input')

        for label in labels:
            generator = self.datagen.flow_from_directory(
                directory=data_dir,
                #     classes=labels'
                classes=[label],
                target_size=(224, 224),
                color_mode="rgb",
                batch_size=32,
                class_mode="categorical",
                shuffle=True,
                seed=42,
                save_to_dir=data_dir / label,
                save_prefix='augmented',
                save_format='jpg'
            )
            all_imgs_per_class = os.listdir(data_dir / label)
            print(label)
            imgs_in_class = len(os.listdir(data_dir / label))
            print(f'Number of imgs in class {label} is {imgs_in_class}.')
            batches_to_generate = (wanted_imgs_cnt_per_class - imgs_in_class)//self.BATCH_SIZE
            if batches_to_generate > 0:
                for t in range(batches_to_generate):
                    generator.__next__()


if __name__ == "__main__":
    preprocessing = Preprocessing()
    preprocessing.image_flow_generator()
