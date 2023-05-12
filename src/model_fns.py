from pathlib import Path
import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# classification report
from sklearn.metrics import classification_report
def prep_finetune_model(model:Model, num_classes:int):
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

    image_generator = gen.flow_from_dataframe(
        dataframe=df,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        color_mode='rgb')
    
    return image_generator


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
