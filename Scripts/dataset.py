from tensorflow import image as image_tf
from tensorflow.data import Dataset
from tensorflow import (constant,
                        float32,
                        io)
from os.path import join
from numpy import array
from os import listdir
from cv2 import imread


def ls(path: str) -> list:
    files = sorted(listdir(path))
    return files


class dataset_model:
    def __init__(self,
                 params: dict) -> None:
        self.params = params
        self._read_dataset()

    def _read_image(self,
                    filename: str,
                    istarget:bool=False) -> array:
        image = io.read_file(filename)
        channel=3
        if istarget:
            channel=1
        image = image_tf.decode_png(image,
                                    channels=channel)
        image = image_tf.convert_image_dtype(image,
                                             float32)
        image = image_tf.resize(image,
                                (self.params["width"],
                                 self.params["height"]))
        image = self._normalization_image(image)
        return image

    def _preprocess_triplets(self,
                             left_image,
                             right_image,
                             target_image) -> array:
        return (
            self._read_image(left_image),
            self._read_image(right_image),
            self._read_image(target_image,
                             istarget=True),
        )

    def _normalization_image(self,
                             image) -> array:
        dos = constant(2,
                       dtype=float32)
        one = constant(1,
                       dtype=float32)
        image = image*dos-one
        return image

    def _read_dataset(self) -> Dataset:
        path = join(self.params["path data"])
        files = ls(path)
        size = len(files)//3
        dataset = self._split_input_target_filenames(path,
                                                     files)
        dataset = self._get_images_dataset(dataset)
        self._split_dataset(dataset,
                            size)

    def _split_dataset(self,
                       dataset:Dataset,
                       size:int)->Dataset:
        dataset=dataset.shuffle(1000000,
                                seed=2022)
        train_size = int(0.9*size)
        test_size = size-train_size
        self.train = dataset.take(train_size)
        self.test = dataset.skip(train_size).take(test_size)

    def _split_input_target_filenames(self,
                                      path: str,
                                      files: list) -> Dataset:
        unique_filenames = ["_".join(file.split("_")[1:])
                            for file in files]
        unique_filenames = set(unique_filenames)
        unique_filenames = list(unique_filenames)
        unique_filenames = sorted(unique_filenames)
        left_filenames = list()
        right_filenames = list()
        target_filenames = list()
        for file in unique_filenames:
            left_image = f"{path}/left_{file}"
            right_image = f"{path}/right_{file}"
            target_image = f"{path}/target_{file}"
            left_filenames.append(left_image)
            right_filenames.append(right_image)
            target_filenames.append(target_image)
        left_filenames = self._convert_to_tensor_dataset(left_filenames)
        right_filenames = self._convert_to_tensor_dataset(right_filenames)
        target_filenames = self._convert_to_tensor_dataset(target_filenames)
        dataset = Dataset.zip((left_filenames,
                               right_filenames,
                               target_filenames))
        return dataset

    def _get_images_dataset(self,
                            dataset: Dataset) -> Dataset:
        dataset = dataset.map(self._preprocess_triplets)
        dataset = dataset.batch(1)
        return dataset

    def _convert_to_tensor_dataset(self,
                                   data: list) -> Dataset:
        dataset = Dataset.from_tensor_slices(data)
        return dataset
