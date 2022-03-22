"""
This file implements the GAN models used in the paper Explanation by Progressive Exaggeration.
This GAN perturbs images to change the output of the model while keeping the image realistic.
The paper can be found here: https://arxiv.org/pdf/1911.00483.pdf
A video explaining the GAN architecture can be found here: https://iclr.cc/virtual_2020/poster_H1xFWgrFPS.html
The GAN architecture is based on the SNGAN. Some code for this architecture can
be found here: https://github.com/henry32144/sngan-projection-notebook/blob/master/CBNGAN_food101.ipynb
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Add, Input, Lambda, Dense, UpSampling2D
from tensorflow.keras import backend as K
from custom_layers import *


def downsampling(input):
    """
    Downsamples input using average pooling. Width and height of input will be halved.
    :param input: Input to be downsampled
    :return: Downsampled input
    """
    return AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(input)


def upsampling(input):
    """
    Upsamples input using nearest interpolation. Width and height of input will be doubled.
    :param input: Input to be upsampled
    :return: Upsampled input
    """
    return UpSampling2D()(input)


class GeneratorModel:
    def __init__(self, input_shape, num_classes):
        """
        :param input_shape: List in format [height, width, channels]. Height and width must be divisible by 32.
        :param num_classes: Number of classes.
        """
        if num_classes < 2:
            raise ValueError("Generator model must have more than 1 possible class.")
        self.input_shape = input_shape
        self.num_classes = num_classes

    def resblock(self, input, in_filters, num_filters, y=None, downsample=True):
        """
        Residual block for the generator model. Downsampling is used in encoder while upsampling is used in decoder.
        :param input: Input to resblock
        :param in_filters: Number of filters in input
        :param num_filters: Number of filters in Conv2D layers
        :param y: Condition for conditional batch normalization. If None, uses regular (unconditional) batch normalization
        :param downsample: Boolean flag. If True, downsamples (dimensions halved), else upsamples (dimensions doubled)
        :return: Output after passing input through resblock
        """
        x = BatchNormalization()(input) if y is None else ConditionalBatchNormalization(self.num_classes, in_filters)([input, y])
        x = Activation("relu")(x)

        if downsample:  # Downsampling x and residual
            x = downsampling(x)
            res = downsampling(input) # Residual
        else:  # Upsampling uses resizing with nearest neighbor interpolation
            x = upsampling(x)
            res = upsampling(input)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x) if y is None else ConditionalBatchNormalization(self.num_classes, num_filters)([x, y])
        x = Activation("relu")(x)
        x = Conv2D(num_filters, 3, padding="same")(x)

        res = Conv2D(num_filters, 1, padding="same")(res)
        return Add()([x, res])

    def get_model(self, num_filters=64):
        """
        This returns the generator model, which is composed of an encoder and decoder and has 2 inputs and 2 outputs.
        Inputs: input (input image) and target (target class of generated image).
        Outputs: encoded (encoded version of input, output of encoder) and generated (new generated image, output of decoder).
        :param num_filters: Number of filters in first encoder layer and last decoder layer.
        :return: Uncompiled model with 2 inputs and outputs
        """
        input = Input(self.input_shape)  # This is the tensor of the original image
        target = Input((1,))  # This is an integer representing the target class
        x = BatchNormalization()(input)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, 3, padding="same")(x)

        # Encoder
        x = self.resblock(x, num_filters, num_filters*2, target)
        x = self.resblock(x, num_filters*2, num_filters*4, target)
        x = self.resblock(x, num_filters*4, num_filters*8, target)
        x = self.resblock(x, num_filters*8, num_filters*16, target)
        encoded = self.resblock(x, num_filters*16, num_filters*16, target)

        # Decoder
        x = self.resblock(encoded, num_filters*16, num_filters*16, target, False)
        x = self.resblock(x, num_filters*16, num_filters*8, target, False)
        x = self.resblock(x, num_filters*8, num_filters*4, target, False)
        x = self.resblock(x, num_filters*4, num_filters*2, target, False)
        x = self.resblock(x, num_filters*2, num_filters, target, False)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(self.input_shape[-1], 3, padding="same")(x)
        generated = Activation("tanh", dtype="float32")(x)  # dtype="float32" necessary for mixed precision
        # generated = Activation("sigmoid", dtype="float32")(x)  # dtype="float32" necessary for mixed precision

        model = Model([input, target], [generated, encoded], name="Generator")
        return model

    @staticmethod
    # @tf.function
    def loss(fake_output):#, target_labels, fake_labels):
        # # TODO: Change this to allow for different num_classes
        # disc_performance = 1. - tf.reduce_mean(fake_output)  # How well generator fooled discriminator
        # target_labels = tf.cast(target_labels, tf.float32) * 0.1
        # fake_labels = tf.cast(fake_labels, tf.float32) * 0.1
        # fake_evaluation = (target_labels * tf.log(fake_labels)) + ((1-target_labels) * tf.log(1-fake_labels))
        # return disc_performance + fake_evaluation
        return 1. - tf.reduce_mean(fake_output)


class DiscriminatorModel:
    def __init__(self, input_shape):
        """
        :param input_shape: List in format [height, width, channels]. Note that height and width must be divisible by 32.
        """
        self.input_shape = input_shape

    @staticmethod
    def get_conv(num_filters, spec_norm=True):
        conv = Conv2D(num_filters, 3, padding="same")
        return SpectralNormalization(conv) if spec_norm else conv

    def resblock(self, input, num_filters, downsample=True, first=False, spec_norm=True):
        """
        Residual block for the discriminator model.
        :param input: Input to resblock
        :param num_filters: Number of filters in Conv2D layers
        :param downsample: Boolean flag. Downsampling occurs if True
        :param first: Boolean flag. If False, adds an extra relu at start. Set to True for first resblock in model.
        :param spec_norm: Boolean flag. If True, spectral normalization applied to convolutional layers.
        :return: Output after passing input through resblock
        """
        if first:  # No initial relu
            x = DiscriminatorModel.get_conv(num_filters, spec_norm)(input)
        else:  # Initial relu
            x = Activation("relu")(input)
            x = DiscriminatorModel.get_conv(num_filters, spec_norm)(x)
        x = Activation("relu")(x)
        x = DiscriminatorModel.get_conv(num_filters, spec_norm)(x)

        if downsample:
            x = downsampling(x)
            input = downsampling(input)  # Residual
            if spec_norm:
                input = SpectralNormalization(Conv2D(num_filters, 1, padding="same"))(input)
            else:
                input = Conv2D(num_filters, 1, padding="same")(input)

        return Add()([x, input])

    def get_model(self, num_filters=64):
        """
        This returns the discriminator model, which determines whether an image is real or generated.
        :param num_filters: Number of filters in first encoder layer and last decoder layer.
        :return: Uncompiled discriminator model
        """
        input = Input(self.input_shape)
        x = self.resblock(input, num_filters, first=True)
        x = self.resblock(x, num_filters*2)
        x = self.resblock(x, num_filters*4)
        x = self.resblock(x, num_filters*8)
        x = self.resblock(x, num_filters*16)
        x = self.resblock(x, num_filters*16, downsample=False)
        x = Activation("relu")(x)
        x = Lambda(lambda z: K.sum(z, axis=[1,2]))(x)  # Global sum pooling, new shape is [None, 1, 1, num_filters*16]
        x = Dense(1)(x)
        x = Activation("sigmoid", dtype="float32")(x)  # dtype="float32" necessary for mixed precision

        model = Model(input, x, name="Discriminator")
        return model

    @staticmethod
    # @tf.function
    def loss(real_output, fake_output):
        """
        Hinge loss function for discriminator.
        :param real_output: Tensor of predictions for real images.
        :param fake_output: Tensor of predictions for fake images.
        """
        # real_loss = tf.reduce_mean(tf.nn.relu(tf.ones_like(real_output) - real_output))
        # fake_loss = tf.reduce_mean(tf.nn.relu(tf.ones_like(fake_output) + fake_output))
        # total_loss = real_loss + fake_loss
        # return total_loss
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


class ExaggerationGan(Model):
    def __init__(self, input_shape, batch_size, pred_model, d_per_g=1, gen_model=None, disc_model=None, print_summary=False):
        """
        :param input_shape: Shape of input images
        :param batch_size: Batch size of GAN
        :param pred_model: Model being explained. Must already be loaded and compiled
        :param d_per_g: Int, number of discriminator steps for each generator step.
        :param gen_model: Generator model. If None, initializes a new generator model with random weights
        :param disc_model: Discriminator model. If None, initializes a new discriminator model with random weights
        :param print_summary: Boolean flag. If True, prints the summary of
        """
        super().__init__()
        self.in_shape = input_shape
        self.batch_size = batch_size
        self.pred_model = pred_model
        # self.file_writer = tf.summary.create_file_writer(logdir)
        self.num_classes = 10  # TODO: Make number of classes variable (remember to also alter bucket calculations)
        self.d_per_g = d_per_g * batch_size  # Multiply by batch size if using enumerated dataset

        self.generator = GeneratorModel(input_shape, self.num_classes).get_model() if gen_model is None else gen_model
        self.discriminator = DiscriminatorModel(input_shape).get_model() if disc_model is None else disc_model
        if print_summary:
            self.generator.summary()
            self.discriminator.summary()

        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    # TODO: Allow for other metrics
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, gen_optimizer, disc_optimizer):
        super().compile()
        self.g_optimizer = gen_optimizer
        self.d_optimizer = disc_optimizer

    # TODO: Test learning rate decay
    def change_g_lr(self, new_lr):
        """
        Changes learning rate of the generator model.
        :param new_lr: New learning rate for generator.
        """
        K.set_value(self.g_optimizer.lr, new_lr)

    def change_d_lr(self, new_lr):
        """
        Changes learning rate of the discriminator model.
        :param new_lr: New learning rate for discriminator.
        """
        K.set_value(self.d_optimizer.lr, new_lr)

    @tf.function
    def train_d_step(self, real_images, target_labels):
        """
        Performs one training step of the discriminator
        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
        :param real_images: 4D tensor of real images.
        :param target_labels: 1D tensor of target labels of generated images. Must have same length as real_images.
        """
        fake_images, _ = self.generator([real_images, target_labels], training=False)  # Generate fake images
        # Open a GradientTape to record the operations run during the forward pass, which enables autodifferentiation.
        with tf.GradientTape(persistent=True) as tape:
            # fake_images, _ = self.generator([real_images, target_labels], training=False)

            real_pred = self.discriminator(real_images, training=True)  # Predictions for real images
            # tf.print(real_pred)
            fake_pred = self.discriminator(fake_images, training=True)  # Predictions for generated images

            d_loss = DiscriminatorModel.loss(real_pred, fake_pred)
        # Calculate the gradients for discriminator
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)

        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        self.disc_loss_tracker.update_state(d_loss)  # Monitor loss
        return d_loss

    @tf.function
    def train_g_step(self, real_images, target_labels):
        """
        Performs one training step of the generator
        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
        :param real_images: 4D tensor of real images.
        :param target_labels: 1D tensor of target labels of generated images. Must have same length as real_images.
        """
        # Open a GradientTape to record the operations run during the forward pass, which enables autodifferentiation.
        with tf.GradientTape(persistent=True) as tape:
            fake_images, _ = self.generator([real_images, target_labels], training=True)
            fake_pred = self.discriminator(fake_images, training=False)
            # fake_labels = tf.math.floor(self.pred_model(fake_images, training=False) * 10.)

            # g_loss = GeneratorModel.loss(fake_pred, target_labels, fake_labels)
            g_loss = GeneratorModel.loss(fake_pred)
        # Calculate the gradients for discriminator
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)

        # Apply the gradients to the optimizer
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.gen_loss_tracker.update_state(g_loss)  # Monitor loss
        # return tf.constant(True)
        return g_loss

    def train_step(self, data):
        """
        :param data: Tuple of 1D tensor containing image numbers and 4D tensor containing images
        """
        image_numbers, real_images = data  # Unpack enumerated data
        # Generate random target labels
        target_labels = tf.random.uniform((self.batch_size,), maxval=self.num_classes - 1, dtype=tf.int32)

        d_loss = self.train_d_step(real_images, target_labels)  # Discriminator training step
        # tf.print(d_loss)

        # Train generator if for any of the images, image_number % self.d_per_g == 0
        train_gen = tf.reduce_any(tf.math.logical_not(tf.cast(image_numbers % self.d_per_g, tf.bool)))
        g_loss = tf.cond(train_gen, lambda: self.train_g_step(real_images, target_labels), lambda: tf.constant(1.))

        # Train generator if discriminator performed well
        # g_loss = tf.cond(d_loss < -.5, lambda: self.train_g_step(real_images, target_labels), lambda: tf.constant(1.))

        return {
            "g_loss_epoch_mean": self.gen_loss_tracker.result(),
            "d_loss_epoch_mean": self.disc_loss_tracker.result(),
            # "g_loss_step": g_loss,
            # "d_loss_step": d_loss,
        }