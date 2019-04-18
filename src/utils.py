# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:17:06 2019

@author: SrivatsanPC
"""

import tensorflow as tf
from curveReader import ImageCompletionReader,GPCurvesReader, PeriodicTSCurvesReader
import numpy as np

# utility methods
def batch_mlp(input, output_sizes, variable_scope):
  """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).
  
  Args:
    input: input tensor of shape [B,n,d_in].
    output_sizes: An iterable containing the output sizes of the MLP as defined 
        in `basic.Linear`.
    variable_scope: String giving the name of the variable scope. If this is set
        to be the same as a previously defined MLP, then the weights are reused.
    
  Returns:
    tensor of shape [B,n,d_out] where d_out=output_sizes[-1]
  """
  # Get the shapes of the input and reshape to parallelise across observations
  batch_size, _, filter_size = input.shape.as_list()
  output = tf.reshape(input, (-1, filter_size))
  output.set_shape((None, filter_size))

  # Pass through MLP
  with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
    for i, size in enumerate(output_sizes[:-1]):
      output = tf.nn.relu(
          tf.layers.dense(output, size, name="layer_{}".format(i)))

    # Last layer without a ReLu
    output = tf.layers.dense(
        output, output_sizes[-1], name="layer_{}".format(i + 1))

  # Bring back into original shape
  output = tf.reshape(output, (batch_size, -1, output_sizes[-1]))
  return output


def get_raw_ts_tensor(train_test_split = 0.8):      
    data = np.random.rand(100,4)
    data = data.astype('float32')
    data[:,0] = [i//10 for i in range(100)]
    
    unique_ids = np.unique(data[:,0])
    random_unique_ids = np.random.permutation(unique_ids)
    split = int(train_test_split * len(random_unique_ids))
    train_ids, test_ids = random_unique_ids[:split], random_unique_ids[split:]
    
      
    train_data = data[np.where(np.isin(data[:,0],train_ids))]
    test_data = data[np.where(np.isin(data[:,0],test_ids))]
 
    return split, len(random_unique_ids)-split, tf.convert_to_tensor(train_data), tf.convert_to_tensor(test_data) 

def get_raw_img_tensor(train_test_split = 0.8, pixel_dim = 1, im_data = 'synthetic'):      
    if im_data == 'synthetic':
        imdata = np.zeros((100*32*32,3+pixel_dim),dtype='float32')
        imdata[:,0] = [i//(32*32) for i in range(100*32*32)]
        imdata[:,1] = [(i/32)%32 for i in range(100*32*32)]
        imdata[:,2] = [i%32 for i in range(100*32*32)]

        unique_ids = np.unique(data[:,0])
        random_unique_ids = np.random.permutation(unique_ids)
        split = int(train_test_split * len(random_unique_ids))
        train_ids, test_ids = random_unique_ids[:split], random_unique_ids[split:]

        train_data = data[np.where(np.isin(data[:,0],train_ids))]
        test_data = data[np.where(np.isin(data[:,0],test_ids))]

        inv_split = len(random_unique_ids)-split
    elif im_data == 'mnist':
        # assert(pixel_dim == 1,'Pixel dim for MNIST should be 1!')

        mnist = tf.keras.datasets.mnist

        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        train_data = np.array([[i,j,k,x_train[i,j,k]] for i in range(x_train.shape[0]) for j in range(x_train.shape[1]) for k in range(x_train.shape[2])],dtype='float32')
        test_data = np.array([[i,j,k,x_test[i,j,k]] for i in range(x_test.shape[0]) for j in range(x_test.shape[1]) for k in range(x_test.shape[2])],dtype='float32')

        split = train_data.shape[0]
        inv_split = test_data.shape[0]

    return split, inv_split, tf.convert_to_tensor(train_data), tf.convert_to_tensor(test_data) 

# Data - expected numpy array - No.of rows by 3 Columns - id, X and Y.
def get_data(data_format, kernel = None, max_context_points = None, 
             random_kernel_parameters = True, test_batch_size =1, train_batch_size =16, min_context_points = None):
    if data_format == 'GP':
        dataset_train = GPCurvesReader(
            batch_size = train_batch_size, max_num_context=max_context_points, 
            random_kernel_parameters=random_kernel_parameters, kernel = kernel)
        data_train = dataset_train.generate_curves()
        
        # Test dataset
        dataset_test = GPCurvesReader(
            batch_size = test_batch_size, max_num_context=max_context_points, testing=True,
            random_kernel_parameters=random_kernel_parameters, kernel = kernel)
        
        data_test = dataset_test.generate_curves()
        
    elif data_format == 'TS':
        train_num_instances, test_num_instances, train_data, test_data = get_raw_ts_tensor()
        
        dataset_train = PeriodicTSCurvesReader(train_batch_size, max_context_points,
                                               train_data, train_num_instances, testing = False)
        data_train = dataset_train.generate_curves()
        
        dataset_test = PeriodicTSCurvesReader(test_batch_size, max_context_points,
                                               test_data, test_num_instances, testing = True)
        
        data_test = dataset_train.generate_curves()

    elif data_format == 'mnist':
        train_num_instances, test_num_instances, train_data, test_data = get_raw_img_tensor(im_data=data_format)
        
        dataset_train = ImageCompletionReader(train_batch_size, min_context_points, max_context_points,
                                               train_data, train_num_instances, testing = False)
        data_train = dataset_train.generate_curves()
        
        dataset_test = ImageCompletionReader(test_batch_size, min_context_points, max_context_points,
                                               test_data, test_num_instances, testing = True)
        
        data_test = dataset_train.generate_curves()

    
    return data_train, data_test
    

