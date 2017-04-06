import tensorflow as tf
import keras
from keras import layers
from keras import activations
from keras import initializers
from keras.layers import Recurrent
from keras import backend as K
from keras.utils import conv_utils
from keras import activations

transpose = layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))
reverse = layers.Lambda(lambda x: tf.reverse(x, [1]))

def SpatialCGRU(x, output_size, **kwargs):
    # Statefully scan the image in each of four directions
    cgru = CGRU(output_size)

    down_rnn = cgru(x)
    up_rnn = reverse(cgru(reverse(x)))
    left_rnn = transpose(cgru(transpose(x)))
    right_rnn = transpose(reverse(cgru(reverse(transpose(x)))))

    concat_out = layers.merge([left_rnn, right_rnn, up_rnn, down_rnn], mode='concat', concat_axis=-1)

    # Convolve the image some more
    output_mask = layers.Conv2D(output_size, (1,1))(concat_out)
    return output_mask


class CGRU(Recurrent):
    """ A 1D convolutional tanh/sigmoid GRU with no dropout and no regularization
    Convolves forward along the first non-batch axis (ie from top to bottom of an image)
    """
    def __init__(self, units=10, *args, **kwargs):
        # __init__ just sets params, doesn't allocate anything
        super(CGRU, self).__init__(return_sequences=True, **kwargs)
        self.units = units

        # TODO: Handle all the normal RNN parameters
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('hard_sigmoid')
        self.dropout = 0
        self.recurrent_dropout = 0

        # TODO: Handle all the normal Conv1D parameters
        self.filter_size = 3
        self.padding = conv_utils.normalize_padding('same')
        # Keras hard-codes the RNN ndim to 3; let's change it to 4
        from keras.engine.topology import InputSpec
        self.input_spec = InputSpec(ndim=4)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.units)

    def build(self, input_shape):
        print("Calling build() for ConvGRU with input shape {}".format(input_shape))
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        pixels_per_step = input_shape[2]
        channels_per_pixel = input_shape[3]

        # Notice that the input_spec changes after __init__ and again after build(), becoming more restrictive
        self.input_spec = keras.engine.InputSpec(shape=(batch_size, time_steps, pixels_per_step, channels_per_pixel))
        self.state_spec = keras.engine.InputSpec(shape=(batch_size, pixels_per_step, self.units))

        # TODO: does Keras require self.states?
        self.states = [None]

        self.Wz = self.add_weight((self.filter_size, self.units, self.units),
                name='Wz', initializer=initializers.get('glorot_uniform'))
        self.Wr = self.add_weight((self.filter_size, self.units, self.units),
                name='Wr', initializer=initializers.get('glorot_uniform'))
        self.Wh = self.add_weight((self.filter_size, self.units, self.units),
                name='Wh', initializer=initializers.get('glorot_uniform'))
        self.Uz = self.add_weight((self.filter_size, channels_per_pixel, self.units),
                name='Uz', initializer=initializers.get('glorot_uniform'))
        self.Ur = self.add_weight((self.filter_size, channels_per_pixel, self.units),
                name='Ur', initializer=initializers.get('glorot_uniform'))
        self.Uh = self.add_weight((self.filter_size, channels_per_pixel, self.units),
                name='Uh', initializer=initializers.get('glorot_uniform'))
        self.built = True

    """ The step() method is a magical function which is passed into tf.while_loop()
        It must have the following special properties:
        Parameters:
            input: tensor with shape `(samples, ...)` (no time dimension),
                representing input for the batch of samples at a certain
                time step.
            states: list of tensors.
        Returns:
            output: tensor with shape `(samples, output_dim)`
                (no time dimension).
            new_states: list of tensors, same length and shapes
                as 'states'. The first state in the list must be the
                output tensor at the previous timestep.
    """
    def step(self, input_tensor, state_tensors):
        # 1 x self.units
        prev_y = state_tensors[0]
        # 1 x self.input_dim
        x_t = input_tensor
        z = self.recurrent_activation(tf.nn.convolution(x_t, self.Uz, padding='SAME') + tf.nn.convolution(prev_y, self.Wz, padding='SAME'))
        r = self.recurrent_activation(tf.nn.convolution(x_t, self.Ur, padding='SAME') + tf.nn.convolution(prev_y, self.Wr, padding='SAME'))
        h = self.activation(
                tf.nn.convolution(x_t, self.Uh, padding='SAME') +
                tf.nn.convolution(prev_y * r, self.Wh, padding='SAME')
            )
        y_t = (1 - z) * h + z * prev_y
        return y_t, [y_t]

    """
    Keras will call this function when you first build the layer.
    """
    def get_initial_states(self, inputs):
        # Our default spatial CGRU will scan the image top to bottom
        #   TODO: Learn to implement a Rule 110 cellular automaton as a demonstration
        # Input shape is BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS
        # Input at each time step is BATCH_SIZE, IMG_WIDTH, CHANNELS
        # Output at each time step is IMG_WIDTH, UNITS 
        # For GRUs, there is no other state but output
        shape = inputs.shape.as_list()
        batch_size, time_steps, pixels_per_step, channels_per_pixel = shape
        initial_state = K.zeros((batch_size, pixels_per_step, self.units))
        return [initial_state]

    def compute_output_shape(self, input_shape):
        batch_size, time_steps, pixels_per_step, channels_per_pixel = input_shape
        return (batch_size, time_steps, pixels_per_step, self.units)

