# utils/model_builders.py
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, Activation, BatchNormalization,
                                     MaxPooling2D, Dropout, Flatten, Dense,
                                     GlobalAveragePooling2D)
from tensorflow.keras import regularizers
from tensorflow.keras.applications import VGG16, ResNet50V2


def build_custom_cnn(input_shape: Tuple[int, int, int] = (48, 48, 1), num_classes: int = 7) -> tf.keras.Model:
    """Builds the custom CNN used in experiments (grayscale input)."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def build_vgg16_tl(input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 7) -> tf.keras.Model:
    """Builds a VGG16-based transfer-learning model."""
    base = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    # Freeze majority of base layers for stability
    for layer in base.layers[:-4]:
        layer.trainable = False

    x = Flatten()(base.output)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=base.input, outputs=output)
    return model


def build_resnet50v2_tl(input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 7) -> tf.keras.Model:
    """Builds a ResNet50V2 based transfer-learning model."""
    base = ResNet50V2(input_shape=input_shape, include_top=False, weights='imagenet')
    # Freeze most layers
    for layer in base.layers[:-50]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    return model
