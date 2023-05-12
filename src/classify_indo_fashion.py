"""
This script classifies the images in the Indo-Fashion dataset using a finetuned pretrained CNN. 

Author: Laura Bock Paulsen (202005791@post.au.dk)
"""
from pathlib import Path
import numpy as np
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16

# local imports 
from model_fns import prep_finetune_model, get_classification_report, prep_data_generator, plot_history

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



def main():
    # SETTING PARAMETERS (maybe move to argsparse with defaults?)
    BATCH_SIZE = 256 * 2
    EPOCHS = 10
    TARGET_SIZE = (224, 224)

    path = Path(__file__)
    
    # load the model
    model = VGG16(include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # load the dataset
    data_path = path.parents[1] / 'data' 
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
    model = prep_finetune_model(model, N_CLASSES)

    # create data generators
    train_generator = prep_data_generator(train_df, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
    val_generator = prep_data_generator(val_df, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
    test_generator = prep_data_generator(test_df, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=False, y_col=None)
    
    # train the model
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, verbose=1)
    
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