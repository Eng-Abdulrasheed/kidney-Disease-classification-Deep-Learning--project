import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,TensorBoard

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config


    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )


    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
            # Removed class_mode='binary' from here
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='binary'  # Moved class_mode here, applicable for both training and validation
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs  # class_mode='binary' is now correctly included here
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,  # Normalize images
            horizontal_flip=True,  # Randomly flip images horizontally (realistic for faces)
            width_shift_range=0.1,  # Randomly translate images horizontally by up to 10%
            height_shift_range=0.1,  # Randomly translate images vertically by up to 10%
            brightness_range=[0.8, 1.2],  # Randomly adjust brightness (80-120% of the original value)
            zoom_range=0.2,  # Randomly zoom in and out on images (80-120% zoom), can be useful for faces
            fill_mode='nearest'  # Strategy to fill newly created pixels, which can appear after a shift or a zoom
            # Avoid using vertical_flip=True for face images
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Callbacks
        checkpoint_path = str(self.config.root_dir / "best_model.h5")
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1,
                mode='min',
                min_delta=0.0001,
                cooldown=0,
                min_lr=0
            ),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]

        # Training
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks  # Add the callbacks here
        )

        # Save the final model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )