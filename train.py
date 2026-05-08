import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix

import config
from utils import create_val_split, check_dataset

tf.random.set_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR,   exist_ok=True)


def build_data_generators():
    create_val_split(config.TRAIN_DIR, config.VAL_DIR)
    check_dataset(config.TRAIN_DIR)

    train_aug = ImageDataGenerator(
        rescale            = 1.0 / 255,
        rotation_range     = 15,
        width_shift_range  = 0.15,
        height_shift_range = 0.15,
        shear_range        = 0.10,
        zoom_range         = 0.15,
        brightness_range   = [0.7, 1.3],
        horizontal_flip    = True,
        fill_mode          = "nearest",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(
        target_size = config.IMG_SIZE,
        batch_size  = config.BATCH_SIZE,
        class_mode  = "binary",
        classes     = config.CLASSES,
        seed        = config.SEED,
    )

    train_gen = train_aug.flow_from_directory(config.TRAIN_DIR, shuffle=True,  **common)
    val_gen   = val_aug  .flow_from_directory(config.VAL_DIR,   shuffle=False, **common)

    print(f"  Class indices : {train_gen.class_indices}")
    print(f"  Train samples : {train_gen.n}")
    print(f"  Val   samples : {val_gen.n}\n")
    return train_gen, val_gen


def conv_block(x, filters, block_name):
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(config.L2),
                      name=f"{block_name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu1")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(config.L2),
                      name=f"{block_name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu2")(x)

    x = layers.MaxPooling2D((2, 2), name=f"{block_name}_pool")(x)
    x = layers.Dropout(0.25, name=f"{block_name}_drop")(x)
    return x


def build_custom_cnn():
    inputs = keras.Input(shape=config.INPUT_SHAPE, name="image_input")

    x = conv_block(inputs, 32,  "block1")
    x = conv_block(x,      64,  "block2")
    x = conv_block(x,      128, "block3")

    x = layers.Conv2D(256, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(config.L2),
                      name="block4_conv1")(x)
    x = layers.BatchNormalization(name="block4_bn1")(x)
    x = layers.Activation("relu", name="block4_relu1")(x)
    x = layers.Conv2D(256, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(config.L2),
                      name="block4_conv2")(x)
    x = layers.BatchNormalization(name="block4_bn2")(x)
    x = layers.Activation("relu", name="block4_relu2")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(config.L2), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(config.DROPOUT, name="fc1_drop")(x)

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(config.L2), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Dropout(config.DROPOUT / 2, name="fc2_drop")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inputs, outputs, name="CustomCNN")


def build_mobilenet(freeze_base=True):
    base = MobileNetV2(input_shape=config.INPUT_SHAPE, include_top=False, weights="imagenet")
    base.trainable = not freeze_base

    inputs = keras.Input(shape=config.INPUT_SHAPE, name="image_input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(config.L2), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(config.DROPOUT, name="fc1_drop")(x)

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(config.L2), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Dropout(config.DROPOUT / 2, name="fc2_drop")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inputs, outputs, name="MobileNetV2_Transfer")


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_callbacks(save_path, log_tag):
    return [
        ModelCheckpoint(filepath=save_path, monitor="val_auc",
                        mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=os.path.join(config.LOG_DIR, log_tag), histogram_freq=1),
    ]


def train_custom_cnn(train_gen, val_gen):
    print("\n" + "=" * 60)
    print("  MODEL A — Custom CNN")
    print("=" * 60)
    model = build_custom_cnn()
    compile_model(model, config.CNN_LR)
    model.summary()

    history = model.fit(
        train_gen,
        epochs           = config.CNN_EPOCHS,
        validation_data  = val_gen,
        callbacks        = get_callbacks(config.CNN_MODEL_PATH, "cnn"),
        steps_per_epoch  = train_gen.n // config.BATCH_SIZE,
        validation_steps = val_gen.n   // config.BATCH_SIZE,
    )
    return history


def train_mobilenet(train_gen, val_gen):
    print("\n" + "=" * 60)
    print("  MODEL B — MobileNetV2 Transfer Learning")
    print("=" * 60)

    print("\n  Phase 1 — Frozen backbone …")
    model = build_mobilenet(freeze_base=True)
    compile_model(model, config.MBNET_INITIAL_LR)
    model.summary()

    h1 = model.fit(
        train_gen,
        epochs           = config.MBNET_EPOCHS_FROZEN,
        validation_data  = val_gen,
        callbacks        = get_callbacks(config.MBNET_SAVE_PATH, "mbnet_p1"),
        steps_per_epoch  = train_gen.n // config.BATCH_SIZE,
        validation_steps = val_gen.n   // config.BATCH_SIZE,
    )

    print("\n  Phase 2 — Fine-tuning last 30 layers …")
    # FIX: use model.layers[1] (the base model object) instead of get_layer() by
    # hardcoded name — the name changes with input size and TF version, causing
    # a ValueError at runtime.
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    compile_model(model, config.MBNET_FINETUNE_LR)

    h2 = model.fit(
        train_gen,
        epochs           = config.MBNET_EPOCHS_FINETUNE,
        validation_data  = val_gen,
        callbacks        = get_callbacks(config.MBNET_SAVE_PATH, "mbnet_p2"),
        steps_per_epoch  = train_gen.n // config.BATCH_SIZE,
        validation_steps = val_gen.n   // config.BATCH_SIZE,
    )
    return h1, h2


def evaluate(model, val_gen, name):
    print(f"\n  ── Evaluating: {name} ──")
    val_gen.reset()
    steps  = val_gen.n // config.BATCH_SIZE + 1
    y_prob = model.predict(val_gen, steps=steps, verbose=1).squeeze()
    y_pred = (y_prob > config.DROWSY_THRESHOLD).astype(int)
    y_true = val_gen.classes[:len(y_pred)]

    print(f"\nClassification Report — {name}:")
    print(classification_report(y_true, y_pred, target_names=config.CLASSES))

    val_gen.reset()
    results = model.evaluate(val_gen, steps=steps, verbose=0)
    return dict(zip(model.metrics_names, results)), y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)  # FIX: use close(fig) instead of show() — prevents blocking in non-GUI environments
    print(f"  Saved → {fname}")


def plot_comparison(cnn_scores, mbnet_scores):
    keys = ["accuracy", "auc", "precision", "recall"]
    x, w = np.arange(len(keys)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, [cnn_scores[k]   for k in keys], w, label="Custom CNN",  color="#4C72B0", alpha=0.85)
    ax.bar(x + w/2, [mbnet_scores[k] for k in keys], w, label="MobileNetV2", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([k.capitalize() for k in keys])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, k in enumerate(keys):
        ax.text(i - w/2, cnn_scores[k]   + 0.01, f"{cnn_scores[k]:.3f}",   ha="center", fontsize=9)
        ax.text(i + w/2, mbnet_scores[k] + 0.01, f"{mbnet_scores[k]:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close(fig)  # FIX: same as above
    print("  Saved → model_comparison.png")


def main():
    print(f"\n  TensorFlow : {tf.__version__}")
    print(f"  GPU        : {bool(tf.config.list_physical_devices('GPU'))}")

    train_gen, val_gen = build_data_generators()
    cnn_hist           = train_custom_cnn(train_gen, val_gen)
    mbnet_h1, mbnet_h2 = train_mobilenet(train_gen, val_gen)

    best_cnn   = keras.models.load_model(config.CNN_MODEL_PATH)
    best_mbnet = keras.models.load_model(config.MBNET_SAVE_PATH)

    cnn_scores,   yt_cnn,   yp_cnn   = evaluate(best_cnn,   val_gen, "Custom CNN")
    mbnet_scores, yt_mbnet, yp_mbnet = evaluate(best_mbnet, val_gen, "MobileNetV2")

    plot_confusion_matrix(yt_cnn,   yp_cnn,   "Custom CNN — Confusion Matrix",  "cm_cnn.png")
    plot_confusion_matrix(yt_mbnet, yp_mbnet, "MobileNetV2 — Confusion Matrix", "cm_mobilenet.png")
    plot_comparison(cnn_scores, mbnet_scores)

    keys = ["accuracy", "auc", "precision", "recall"]
    print("\n" + "=" * 60)
    print(f"  {'Metric':<15} {'Custom CNN':>12} {'MobileNetV2':>14}")
    print("  " + "-" * 42)
    for k in keys:
        print(f"  {k:<15} {cnn_scores[k]:>12.4f} {mbnet_scores[k]:>14.4f}")
    print("=" * 60)

    winner = "MobileNetV2" if mbnet_scores["auc"] >= cnn_scores["auc"] else "Custom CNN"
    print(f"\n  Best model by AUC → {winner}")
    print(f"  Models saved to   → {config.MODEL_DIR}\n")


if __name__ == "__main__":
    main()
