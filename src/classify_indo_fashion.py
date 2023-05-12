"""
This script classifies the images in the Indo-Fashion dataset using a finetuned pretrained CNN. 

Author: Laura Bock Paulsen (202005791@post.au.dk)
"""

from pathlib import Path
import numpy as np

# tf tools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# VGG16 model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt

import pickle
import json
import pandas as pd


def dataframe_from_json(json_path:Path):
    """
    Create a pandas dataframe from a json file.

    Parameters
    ----------
    json_path : Path
        Path to the json file containing the data.

    Returns
    -------
    df : DataFrame
        Pandas dataframe.
    """
    
    data = []

    # Read json file by loading each line as a dictionary
    with open(json_path) as f:
        for line in f:
            data.append(json.loads(line))

    return pd.DataFrame(data)

def prep_finetuning_model(model:Model, num_classes:int):
    """
    Prepares a pretrained model for finetuning by freezing the layers and adding new classification layers.

    Parameters
    ----------
    model : Model
        Pretrained model.
    num_classes : int
        Number of classes.
    
    Returns
    -------
    model : Model
        Model ready for finetuning with new classification layers.
    """

    # load model without the top layer
    model = VGG16(include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # freeze the layers
    for layer in model.layers:
        layer.trainable = False

    # add new classification layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(num_classes, activation='softmax')(class1)

    # define new model
    model = Model(inputs=model.inputs, outputs=output)


    # compile model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,
                                                                 decay_steps=10000,
                                                                 decay_rate=0.9)
    
    opt = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def prep_data_generator(df:pd.DataFrame, batch_size:int, target_size:tuple, x_col='image_path', y_col='class_label', shuffle:bool = True):
    """
    Prepares data generators for training, validation and testing.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the training data.
    batch_size : int
        Batch size.
    target_size : tuple
        Target size for the images.
    x_col : str
        Name of column with image paths
    y_col : str or None
        Name of column with class labels
   shuffle : bool
        Whether to shuffle the images.

    Returns
    -------
    image_generator
        Data generator.
    """
    # create data generator
    gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    image_generator = train_gen.flow_from_dataframe(
        dataframe=df,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        color_mode='rgb')
    
    return image_generator

def plot_history(history, save_path:Path=None):
    """
    Plots the training history.

    Parameters
    ----------
    history : History
        Training history.
    save_path : Path, optional
        Path to save the plot to, by default None
    
    Returns
    -------
    fig : Figure
        Figure object.
    axes : Axes
        Axes object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi = 300, sharex=True)

    # plot accuracy
    axes[0].plot(history.history["accuracy"], label='train')
    axes[0].plot(history.history["val_accuracy"],linestyle = "--",label="val")
    axes[0].set_title("Accuracy")

    # plot loss
    axes[1].plot(history.history['loss'], label='train')
    axes[1].plot(history.history['val_loss'], label='val', linestyle = "--")
    axes[1].set_title('Loss')

    # add legend
    axes[0].legend()
    axes[1].legend()

    # add labels
    fig.supxlabel('Epoch')

    plt.tight_layout()

    # save plot
    if save_path:
        plt.savefig(save_path)
    
    return fig, axes

def get_classification_report(y_true, y_pred, target_names, save_path:Path):
    """
    Gets the classification report and saves it to a txt file if a path is provided.

    Parameters
    ----------
    y_true : array
        True labels.
    y_pred : array
        Predicted labels.
    target_names : list
        List of target names.
    save_path : Path
        Path to save the report to. If None, the report is not saved.

    Returns
    -------
    report : 
        Classification report.

    """
    # get classification report
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    # save report
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(report.encode('utf-8'))
    
    return report


def main():

    # SETTING PARAMETERS (maybe move to argsparse with defaults?)

    BATCH_SIZE = 256 * 2
    EPOCHS = 10
    TARGET_SIZE = (224, 224)

    path = Path(__file__)
    
    # load the model
    model = VGG16()

    # load the dataset
    data_path = Path("/work/431824")   #  path.parents[1] / 'data' '
    meta_path = data_path /'images' / 'metadata'

    # load in the labels
    train_df = dataframe_from_json(meta_path  / 'train_data.json')
    val_df = dataframe_from_json(meta_path / 'val_data.json')
    test_df = dataframe_from_json(meta_path / 'test_data.json')

    N_CLASSES = train_df['class_label'].nunique()

    # add full path to image
    for data in [train_df, val_df, test_df]:
        data['image_path'] = data['image_path'].apply(lambda x: str(data_path / x))

    # prep model for finetuning
    model = prep_finetuning_model(model, N_CLASSES)

    # create data generators
    train_generator = prep_data_generator(train_df, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
    val_generator = prep_data_generator(val_df, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
    test_generator = prep_data_generator(test_df, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=False, y_col=None)
    
    # train the model
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        verbose=1)
    
    # plot training history
    plot_path = path.parents[1] / 'mdl_reports' / 'history.png'
    fig, axes = plot_history(history, save_path = plot_path)

    # save history as pickle
    history_path = path.parents[1] / 'mdl_reports' / 'history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    # get predictions
    y_pred = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)

    # classification report
    report = get_classification_report(test_generator.classes, y_pred, test_generator.class_indices, save_path = path.parents[1] / 'mdl_reports' / 'report.txt')

    # save model
    model.save("mdl")

if __name__ == "__main__":
    main()