'''
This file contains custom layers used by the progressive exaggeration GAN model.
'''
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Embedding, BatchNormalization
from tensorflow.keras import backend as K


# TODO: Adjust this layer so it works for variable batch size so it doesn't throw errors at epoch end
class ConditionalBatchNormalization(tf.keras.Model):
    """
    https://github.com/crcrpar/pytorch.sngan_projection/blob/master/links/conditional_batchnorm.py
    Conditional Batch Normalization Base Class
    """

    def __init__(self, num_classes, units, **kwargs):
        """
        Initializes Embedding for use in weights and biases of batch normalization
        :param num_classes: Number of possible classes
        :param units: Number of channels in output
        """
        super().__init__(**kwargs)
        self.weight_embedding = Embedding(input_dim=num_classes, output_dim=units, input_length=1,
                                          embeddings_initializer='ones')
        self.bias_embedding = Embedding(input_dim=num_classes, output_dim=units, input_length=1,
                                        embeddings_initializer='zeros')
        self.bn = BatchNormalization()

    def call(self, inputs, training=None):
        """
        This function performs batch normalization when the layer is called
        :param inputs: (input, class) tuple consisting of input to bn and class numbed respectively
        :param training: Input to training parameter of batch normalization
        """
        # Get class weights and bias for the class number provided
        class_weights = self.weight_embedding(inputs[1])
        class_bias = self.bias_embedding(inputs[1])

        # Ensure class_weights and class_bias are 2D tensors
        if len(class_weights.shape) == 3:
            class_weights = tf.squeeze(class_weights, [1])
        if len(class_bias.shape) == 3:
            class_bias = tf.squeeze(class_bias, [1])

        output = self.bn(inputs[0], training=training)

        # Ensure class_weights and class_bias are 2D tensors
        if len(class_weights.shape) == 1:
            class_weights = tf.expand_dims(class_weights, 0)
        if len(class_bias.shape) == 1:
            class_bias = tf.expand_dims(class_bias, 0)

        # Expand class_weights and class_bias to 4D tensors
        class_weights = class_weights[:, tf.newaxis, tf.newaxis, :]
        class_bias = class_bias[:, tf.newaxis, tf.newaxis, :]

        # Weight and bias input
        output = class_weights * output + class_bias
        return output


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Reference from WeightNormalization implementation of TF Addons
    SpectralNormalization wrapper works for keras CNN and Dense (RNN not tested).
    ```python
      net = SpectralNormalization(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3))(x)
    ```
    Arguments:
      layer: a layer instance.
    Raises:
      ValueError: If `Layer` does not contain a `kernel` of weights
    """

    def __init__(self, layer, power_iter=1, **kwargs):
        super().__init__(layer, **kwargs)
        self._track_trackable(layer, name='layer')
        self.power_iter = 1
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)
        self.is_embedding = isinstance(self.layer, tf.keras.layers.Embedding)

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        if self.is_rnn:
            kernel_layer = self.layer.cell
        else:
            kernel_layer = self.layer

        if not hasattr(kernel_layer, 'kernel') and not hasattr(kernel_layer, 'embeddings'):
            raise ValueError('`SpectralNormalization` must wrap a layer that'
                             ' contains a `kernel or embedding` for weights')

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        elif self.is_embedding:
            kernel = kernel_layer.embeddings
        else:
            kernel = kernel_layer.kernel

        self.kernel_shape = kernel.shape

        self.u = self.add_weight(name='u',
                             shape=(self.kernel_shape[-1], 1),
                             initializer='random_normal',
                             trainable=False)
        self.w = kernel
        self.built = True

    def call(self, inputs, training=True):
        """Call `Layer`"""

        with tf.name_scope('compute_weights'):

            weight = tf.reshape(self.w, shape=[self.kernel_shape[-1], -1])
            # power iteration
            for i in range(self.power_iter):
                v = K.l2_normalize(tf.matmul(tf.transpose(weight), self.u))
                u = K.l2_normalize(tf.matmul(weight, v))
            v = tf.stop_gradient(v)
            u = tf.stop_gradient(u)

            sigma = tf.matmul(tf.matmul(tf.transpose(u), weight), v)
            kernel = self.w / sigma

            if self.is_rnn:
                print(self.is_rnn)
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
