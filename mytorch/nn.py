import numpy as np

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution
    Returns:
        int: size of the output as an int
    """
    # For use in Conv1d.forward() using the formula in the main notebook
    # Uses floor division
    return (input_size - kernel_size) // stride + 1
    
class Conv1d:
    """1-dimensional convolutional layer.
    See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html for explanations
    and ideas.

    Args:
        in_channel (int): # channels in input (example: # color channels in image)
        out_channel (int): # channels produced by layer
        kernel_size (int): edge length of the kernel (i.e. 3x3 kernel <-> kernel_size = 3)
        stride (int): Stride of the convolution (filter)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Randomly initializing weight/bias (Kaiming uniform)
        bound = np.sqrt(1 / (in_channels * kernel_size))
        self.weight = np.random.normal(-bound, bound, size=(out_channels, in_channels, kernel_size))
        self.bias = np.random.normal(-bound, bound, size=(out_channels,))

        self.grad_weight = np.zeros(self.weight.shape)
        self.grad_bias = np.zeros(self.bias.shape)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channels, input_size)
        Return:
            out (np.array): (batch_size, out_channels, output_size)
        """
        # [Given] For your convenience, got the shape variables and stored input
        batch_size, _, input_size = x.shape
        self.x = x
        
        # [Given] Store input and pre-calculate the number of output slices
        self.output_size = get_conv1d_output_size(input_size, self.kernel_size, self.stride)

        # Declare output array filled with zeros of appropriate shape
        out = np.zeros((batch_size, self.out_channels, self.output_size))

        # Implement the pseudocode given in the main notebook
        """
        Pseudocode
            [Given] Pre-calculate and store the number of output slices
            Pre-declare an output array filled with zeros, shaped (batch_size, out_channels, output_size)
                This is where we'll store the results of each tensordot
            For each output slice to calculate:
                Determine the beginning/end index of the current input slice
                    Hint: beg_index = i * stride, what is end_index?
                Do a tensordot between the weight matrix and the 2nd/3rd axes of the input
                Add the bias to the result
                Store the result in the appropriate slice of the output array (hint: index along the last axis of out)
            Return the output array
        """
        beg_index = 0
        end_index = 0
        # for each output slice
        for i in range(self.output_size):
            beg_index = i * self.stride
            end_index = beg_index + self.kernel_size
            out[:, :, i] = np.tensordot(self.weight, self.x[:, :, beg_index:end_index], axes=([1, 2], [1, 2])).T + self.bias
            """
            Step by step explanation of the line:
                1. self.x[:, :, beg_index:end_index] - Extracts a sliding window of the input with 
                    shape (batch_size, in_channels, kernel_size)
                2. np.tensordot(self.weight, ..., axes=([1, 2], [1, 2])) - Performs a tensor dot product that:
                    * Takes self.weight with shape (out_channels, in_channels, kernel_size)
                    * Contracts axes [1, 2] of weight with axes [1, 2] of the input slice
                    * This effectively applies each of the out_channels filters to the input window
                    * Results in shape (out_channels, batch_size)
                3. .T - Transposes the result from (out_channels, batch_size) to (batch_size, out_channels)
                4. + self.bias - Adds the bias vector (out_channels,) to produce the final 
                    result with shape (batch_size, out_channels)
                5. out[:, :, i] = - Stores this result in the i-th position of the output tensor
            In summary: It slides the kernel across the input, applies each filter (via tensordot), 
            transposes to match output dimensions, adds bias, and stores the result.
            """
        return out
        
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channels, output_size)
        Return:
            dx (np.array): (batch_size, in_channels, input_size)
        """
        # [Given] Initialize the gradient of the input to zeros
        dx = np.zeros(self.x.shape) # all gradients are the same shape as their originals
        
        # Follow pseudocode, calculating influences and adding them to correct positions
        """
        Pseudocode:
            1) Make array filled with 0's, the same shape as the original input x (Given)
            2) For each slice in the number of output slices:
                (i) Calculate the beginning/end index of our current slice, exactly like we did in forward()
                (ii) Add to dx[:,:,b:e] using the += operator:
                    tensordot between delta[:,:,i] and self.weight along axes=(1)
                    * In other words, we're accumulating the influence of slice i of the output 
                      on indices [b:e] of the input. 
                    * We use a += because some slices of the input may have been used multiple times, 
                      so we just sum up their influences
                (iii) Add to self.grad_weight using the += operator:
                    tensordot between delta[:,:,i].T and self.x[:,:,b:e] along axes=(1)
                    * We add to the entire weight's gradient because the entire kernel was used for this slice
                    * Again, it's += because the kernel possibly saw parts of the input multiple 
                      times, so we just sum up its total influence on those parts
            3) We can calculate the bias's gradient in one line of code:
                (i) Set self.grad_bias equal to the sum of delta along axes (0,2)
                (ii) This works because the bias affected each part of the output the same way, so we just need its total influence.
        """
        beg_index = 0
        end_index = 0
        # for each output slice
        for i in range(self.output_size):
            beg_index = i * self.stride
            end_index = beg_index + self.kernel_size
            dx[:, :, beg_index:end_index] += np.tensordot(delta[:, :, i], self.weight, axes=([1], [0]))
            self.grad_weight += np.tensordot(delta[:, :, i].T, self.x[:, :, beg_index:end_index], axes=([1], [0]))

        self.grad_bias = np.sum(delta, axis=(0, 2))
        return dx