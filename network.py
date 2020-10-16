import tensorflow as tf
from datasets import IMG_SHAPE
from loss import *

def get_MobileNet_backbone_network(embedding_size=128, fc_layer_size=512, l2_norm=True, pretrained=True, trainable_base=False, dropout=False):
    if pretrained:
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
    else:
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='None')
    base_model.trainable = trainable_base

    if dropout:
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(fc_layer_size, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embedding_size), ])
    else:
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(fc_layer_size, activation='relu'),
            tf.keras.layers.Dense(embedding_size), ])
    if l2_norm:
        model.add(tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1)))

    return model


def get_ResNet_backbone_network(embedding_size=128, fc_layer_size=1024, l2_norm=True, pretrained=True, trainable_base=False, dropout=False):
    if pretrained:
        base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')
    else:
        base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='None')
    base_model.trainable = trainable_base

    if dropout:
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(fc_layer_size, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embedding_size), ])
    else:
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(fc_layer_size, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(embedding_size), ])
    if l2_norm:
        model.add(tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1)))

    return model


@tf.function
def train_step(X, y, model, optimizer, margin=1):
    with tf.GradientTape() as tape:
        embeddings = model(X, training=True)
        loss, fraction = batch_all_triplet_loss(y, embeddings, margin)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, fraction


@tf.function
def val_step(X, y, model, margin=1):
    embeddings = model(X)
    loss, fraction = batch_all_triplet_loss(y, embeddings, margin)
    return loss, fraction
