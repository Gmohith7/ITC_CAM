"""
Convert a trained Keras .h5 model to a quantized TFLite model.
Run on PC after training.

Usage:
    python training/export_tflite.py --model trained_model.h5 --output model/model.tflite
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="trained_model.h5")
    p.add_argument("--output", default="model/model.tflite")
    p.add_argument("--quantize", action="store_true", default=True,
                   help="Apply int8 post-training quantization (default: on)")
    return p.parse_args()


def main():
    args = parse_args()
    import tensorflow as tf
    import os

    model = tf.keras.models.load_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[Export] Quantization enabled (int8 post-training).")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"[Export] TFLite model saved → {args.output}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
