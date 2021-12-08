import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds


class MNISTModel:

    def __init__(self, weights: str = None):
        self.model = self.get_model()
        if weights:
            self.model.load_weights(weights)
            self.model.build(input_shape=(None, 28, 28, 1))
            self.model.summary()

    def get_model(self) -> keras.models.Sequential:
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(28, (3, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(10, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['sparse_categorical_accuracy'])

        return model

    def train(self, epochs=10) -> None:
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label

        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        loss_history = []
        for i in range(epochs):
            epoch_result = self.model.fit(ds_train, validation_data=ds_test)

            loss_history.append(epoch_result.history['loss'])
            if min(loss_history) == epoch_result.history['loss']:
                print('Minimal loss so far: ', min(loss_history))
                self.model.save_weights('model/weights.hdf')

        self.model.summary()
        self.model.evaluate(ds_test)

    def classify_number(self, img) -> np.ndarray:
        """
        Generates the given amount of sequences
        """
        img = np.reshape(img, (1, 28, 28))
        result = self.model.predict(img)
        result = np.argmax(result)

        return result
