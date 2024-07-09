import tensorflow as tf


class ReduceColumnsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReduceColumnsLayer, self).__init__(**kwargs)

    def call(self, inputs):
        tensor, C = inputs
        B, N, M, K = tf.unstack(tf.shape(tensor))

        # Squeeze C and ensure it is an integer tensor
        C = tf.cast(tf.squeeze(C), tf.int32)

        # Calculate the number of columns to reduce
        remaining_cols = M - C
        num_pairs = remaining_cols // 2

        # If remaining_cols is odd, include the last column in the averaging process
        if remaining_cols % 2 != 0:
            num_pairs += 1

        # Reshape to split into pairs of columns for the remaining columns
        reshaped = tf.reshape(tensor[:, :, :num_pairs * 2, :], (B, N, num_pairs, 2, K))

        # Take the average of pairs of columns
        averaged = tf.reduce_mean(reshaped, axis=3)

        # Concatenate the averaged columns with the last (M - num_pairs * 2) columns
        fixed_cols = tensor[:, :, num_pairs * 2:, :]
        output = tf.concat([averaged, fixed_cols], axis=2)

        return output


# Example usage inside a tf.function
@tf.function
def example_usage(input_tensor, C):
    # Create the layer
    reduce_columns_layer = ReduceColumnsLayer()
    # Apply the layer
    output = reduce_columns_layer([input_tensor, C])
    return output


B, N, M, K = 4, 6, 128, 3  # Sample dimensions
C = 10  # Specify C as an integer value
# Create a sample input tensor and C tensor
input_tensor = tf.random.normal((B, N, M, K))

for idx in range(20):
    output_tensor = example_usage(input_tensor, C)
    print("Output shape:", output_tensor.shape)
    input_tensor = output_tensor
