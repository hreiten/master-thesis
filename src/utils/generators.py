import numpy as np 

"""
Defines generators that generates batches of data for the deep learning models. 
"""

def batch_generator(x_data, y_data, lookback, batch_size=128):
    """
    Generator function for creating random batches of training-data.
    
    :param numpy.ndarray x_data: Numpy array of the features, not targets. 2D array, normalized/scaled, numpy.
    :param numpy.ndarray y_data: Numpy array of the targets, not features. 2D array, normalized/scaled, numpy. 
    :param int lookback: How many timesteps back the input data should go.
    :param int batch_size: The number of samples per batch. Default=128. 
    
    :return tuple (x_batch, y_batch): Batch of samples and corresponding targets.
    """
    num_x_signals = x_data.shape[1]
    num_y_signals = y_data.shape[1]

    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, lookback, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, lookback, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(x_data.shape[0] - lookback)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_data[idx:idx+lookback]
            y_batch[i] = y_data[idx:idx+lookback]
        
        yield (x_batch, y_batch)
        
        
def chollet_generator(x_data, y_data, lookback=200, min_index=0, max_index=None,
                      shuffle=False, batch_size=128, step=1):
    """
    Generator function for creating batches of training-data.
    This is the same generator as proposed by François Chollet in "Deep Learning with Python".
    
    :param numpy.ndarray x_data: Numpy array of the samples. Numpy 2D array, normalized/scaled. 
    :param numpy.ndarray y_data: Numpy array of the targets. Numpy 2D array, normalized/scaled. 
    :param int lookback: How many timesteps back the input data should go. 
    :param int min_index & max_index: Indices in the data array that delimit which timesteps to draw from. 
                                      This is useful for splitting the data. 
    :param bool shuffle: Whether to shuffle our samples or draw them in chronological order.
    :param int batch_size: The number of samples per batch.
    :param int step: The period, in timesteps, at which we sample data.
    
    :return tuple (samples, targets): Batches of samples and targets. 
    """
    
    if max_index is None:
        max_index = len(x_data) - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows),
                           lookback // step,
                           x_data.shape[-1]))
        targets = np.zeros((len(rows),
                            y_data.shape[-1]))
        
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            indices_targets = rows[j]
            
            samples[j] = x_data[indices]
            targets[j] = y_data[indices_targets]
        yield samples, targets
        
