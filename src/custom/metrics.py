import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils, generic_utils
from tensorflow.keras import backend


class Specificity(Metric):
    '''
    Computes the specificity of the predictions with respect to the labels. The metric creates two local variables,
    `true_negatives` and `false_positives` that are used to compute the specificity. This value is ultimately returned as
    `specificity`, an idempotent operation that simply divides `true_negatives` by the sum of `true_negatives` and
    `false_positives`. If `sample_weight` is `None`, weights default to 1. Use `sample_weight` of 0 to mask values.
    If `top_k` is set, we'll calculate specificity as how often on average a class among the top-k classes with the
    highest predicted values of a batch entry is correct and can be found in the label for that entry. If `class_id` is
    specified, we calculate specificity by considering only the entries in the batch for which `class_id` is above the
    threshold and/or in the top-k highest predictions, and computing the fraction of them for which `class_id` is indeed
    a correct label.
    '''

    def __init__(self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        '''
        :param thresholds: (Optional) A float value or a python list/tuple of float threshold values in [0, 1]. A
            threshold is compared with prediction values to determine the truth value of predictions (i.e., above the
            threshold is `true`, below is `false`). One metric value is generated for each threshold value. If neither
            thresholds nor top_k are set, the default is to calculate specificity with `thresholds=0.5`.
        :param top_k: (Optional) Unset by default. An int value specifying the top-k predictions to consider when
            calculating specificity.
        :param class_id: (Optional) Integer class ID for which we want binary metrics. This must be in the half-open
        interval `[0, num_classes)`, where `num_classes` is the last dimension of predictions.
        :param name: (Optional) string name of the metric instance.
        :param dtype: (Optional) data type of the metric result.
        '''

        super(Specificity, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        Accumulates true negative and false positive statistics.
        :param y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
        :param y_pred: The predicted values. Each element must be in the range `[0, 1]`.
        :param sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        :returns: Update op.
        '''
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        result = tf.math.divide_no_nan(self.true_negatives, self.true_negatives + self.false_positives)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(generic_utils.to_list(self.thresholds))
        backend.batch_set_value([(v, np.zeros((num_thresholds,)))
                                 for v in (self.true_negatives,
                                           self.false_positives)])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(Specificity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))