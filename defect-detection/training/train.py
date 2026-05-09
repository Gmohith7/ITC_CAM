"""
Train a MobileNetV2-based binary classifier (OK / DEFECT).
Run this on a PC or Google Colab — NOT on the Raspberry Pi.

Usage:
    python training/train.py --data data/annotated --epochs 20 --output trained_model.h5
"""

import argparse
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/annotated")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--output", default="trained_model.h5")
    return p.parse_args()


def main():
    args = parse_args()

    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    IMG_SIZE = (224, 224)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
    )

    train_gen = datagen.flow_from_directory(
        args.data, target_size=IMG_SIZE, batch_size=args.batch,
        class_mode="categorical", subset="training"
    )
    val_gen = datagen.flow_from_directory(
        args.data, target_size=IMG_SIZE, batch_size=args.batch,
        class_mode="categorical", subset="validation"
    )

    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(train_gen.num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True),
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)
    model.save(args.output)
    print(f"[Train] Saved → {args.output}")

    # Print class index map so labels.txt can be verified
    print("[Train] Class indices:", train_gen.class_indices)


if __name__ == "__main__":
    main()
