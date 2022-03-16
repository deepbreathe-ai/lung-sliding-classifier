'''
Script for running predictions using trained models
'''

import datetime
import pandas as pd
from src.visualization.visualization import *
from src.models.models import get_model
from preprocessor import MModePreprocessor
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, \
    FalsePositives, Accuracy
from custom.metrics import Specificity

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def predict_set(model, test_df, threshold=0.5, model_def_str=cfg['TRAIN']['MODEL_DEF']):
    '''
    Given a dataset, make predictions for each constituent example.
    :param model: A trained TensorFlow model
    :param test_df: DataFrame containing npz files to predict
    :param threshold: Classification threshold
    :param model_def_str: Model name
    :return: List of predicted classes, array of prediction probabilities
    '''

    # Get the model function and preprocessing function
    model_def_fn, preprocessing_fn = get_model(model_def_str)

    # Create TF datasets for training, validation and test sets
    # Note: This does NOT load the dataset into memory! We specify paths,
    #       and labels so that TensorFlow can perform dynamic loading.
    dataset = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label']))
    preprocessor = MModePreprocessor(preprocessing_fn)

    preprocessed_set = preprocessor.prepare(dataset, test_df, shuffle=False, augment=False)
    # Obtain prediction probabilities and classes
    p = model.predict(preprocessed_set, verbose=1)
    pred_classes = (p[:, 0] >= threshold).astype(int)

    return pred_classes, np.squeeze(p, axis=1)


def export_predictions(df, pred_classes, probs):
    '''
    Export prediction outputs to csv
    :param df: DataFrame of npz files
    :param pred_classes: List of predicted classes
    :param probs: List of probabilities
    '''

    pred_df = pd.DataFrame(columns=['id', 'Predicted Class', 'Probability'])
    pred_df['id'] = df['id']
    pred_df['Ground Truth'] = df['label']
    pred_df['Predicted Class'] = pred_classes
    pred_df['Probability'] = probs
    pred_df.to_csv(os.path.join(os.getcwd(),
                                cfg['PREDICT']['PREDICTIONS_OUT'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                + '_predictions.csv'), index=False)

def export_metrics(df, pred_classes):
    metrics = [Accuracy, AUC, Recall, Specificity, Precision, TrueNegatives, TruePositives, FalseNegatives, FalsePositives]
    results = []
    metric_names = []
    for metric in metrics:
        m = metric()
        m.update_state(df['label'], pred_classes)
        results += [m.result().numpy()]
        metric_names += [m.name]
    res_df = pd.DataFrame([results], columns=metric_names)
    res_df.to_csv(os.path.join(os.getcwd(),
                                cfg['PREDICT']['PREDICTIONS_OUT'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                + '_metrics.csv'), index=False)

if __name__ == '__main__':
    test_df = pd.read_csv(cfg['PREDICT']['TEST_DF'])
    model = load_model(os.path.join('..', cfg['PREDICT']['MODEL']), compile=False)
    pred_labels, probs = predict_set(model, test_df)
    export_predictions(test_df, pred_labels, probs)
    export_metrics(test_df, pred_labels)
