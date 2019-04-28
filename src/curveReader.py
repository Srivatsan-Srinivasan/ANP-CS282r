# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:56:38 2019

@author: SrivatsanPC
"""
import collections
import tensorflow as tf

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


class GPCurvesReader(object):
  """Generates curves using a Gaussian Process (GP).

  Supports vector inputs (x) and vector outputs (y). Kernel is
  mean-squared exponential, using the x-value l2 coordinate distance scaled by
  some factor chosen randomly in a range. Outputs are independent gaussian
  processes.
  """

  def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               l1_scale=0.6,
               sigma_scale=1.0,
               random_kernel_parameters=True,
               kernel = 'SE', #valid options {SE,PER}
               testing=False):
    """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      x_size: Integer >= 1 for length of "x values" vector.
      y_size: Integer >= 1 for length of "y values" vector.
      l1_scale: Float; typical scale for kernel distance function.
      sigma_scale: Float; typical scale for variance.
      random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma) 
          will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
    """
    self._batch_size = batch_size
    self._max_num_context = max_num_context
    self._x_size = x_size
    self._y_size = y_size
    self._l1_scale = l1_scale
    self._sigma_scale = sigma_scale
    self._random_kernel_parameters = random_kernel_parameters
    self._testing = testing
    self._kernel = kernel
    
  def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
    """Applies the Gaussian kernel to generate curve data.

    Args:
      xdata: Tensor of shape [B, num_total_points, x_size] with
          the values of the x-axis data.
      l1: Tensor of shape [B, y_size, x_size], the scale
          parameter of the Gaussian kernel.
      sigma_f: Tensor of shape [B, y_size], the magnitude
          of the std.
      sigma_noise: Float, std of the noise that we add for stability.

    Returns:
      The kernel, a float tensor of shape
      [B, y_size, num_total_points, num_total_points].
    """
    num_total_points = tf.shape(xdata)[1]

    # Expand and take the difference
    xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
    xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
    diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

    # [B, y_size, num_total_points, num_total_points, x_size]
    if self._kernel == 'PER':
        norm = 2*tf.square(tf.math.sin(3.14*diff[:, None, :, :, :])) / l1[:, :, None, None, :]
        norm = tf.reduce_sum(norm, -1)  # [B, data_size, num_total_points, num_total_points]
        # [B, y_size, num_total_points, num_total_points]
        kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-norm)

    else: # if kernel is normal gaussian
        norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
        norm = tf.reduce_sum(norm, -1)  # [B, data_size, num_total_points, num_total_points]
        # [B, y_size, num_total_points, num_total_points]
        kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5*norm)

    # Add some noise to the diagonal to make the cholesky work.
    kernel += (sigma_noise**2) * tf.eye(num_total_points)

    return kernel

  def generate_curves(self, seed = None):
    """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
    """
    num_context = tf.random_uniform(
        shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32, seed=seed)

    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.
    if self._testing:
      num_target = 400
      num_total_points = num_target
      x_values = tf.tile(
          tf.expand_dims(tf.range(-2., 2., 1. / 100, dtype=tf.float32), axis=0),
          [self._batch_size, 1])
      x_values = tf.expand_dims(x_values, axis=-1)
    # During training the number of target points and their x-positions are
    # selected at random
    else:
      num_target = tf.random_uniform(shape=(), minval=0, 
                                     maxval=self._max_num_context - num_context,
                                     dtype=tf.int32, seed=seed)
      num_total_points = num_context + num_target
      x_values = tf.random_uniform(
          [self._batch_size, num_total_points, self._x_size], -2, 2, seed=seed)

    # Set kernel parameters
    # Either choose a set of random parameters for the mini-batch
    if self._random_kernel_parameters:
      l1 = tf.random_uniform([self._batch_size, self._y_size,
                              self._x_size], 0.1, self._l1_scale)
      sigma_f = tf.random_uniform([self._batch_size, self._y_size],
                                  0.1, self._sigma_scale)
    # Or use the same fixed parameters for all mini-batches
    else:
      l1 = tf.ones(shape=[self._batch_size, self._y_size,
                          self._x_size]) * self._l1_scale
      sigma_f = tf.ones(shape=[self._batch_size,
                               self._y_size]) * self._sigma_scale

    # Pass the x_values through the Gaussian kernel
    # [batch_size, y_size, num_total_points, num_total_points]
    kernel = self._gaussian_kernel(x_values, l1, sigma_f)

    # Calculate Cholesky, using double precision for better stability:
    cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

    # Sample a curve
    # [batch_size, y_size, num_total_points, 1]
    y_values = tf.matmul(
        cholesky,
        tf.random_normal([self._batch_size, self._y_size, num_total_points, 1], seed=seed))

    # [batch_size, num_total_points, y_size]
    y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

    if self._testing:
      # Select the targets
      target_x = x_values
      target_y = y_values

      # Select the observations
      idx = tf.random_shuffle(tf.range(num_target), seed=seed)
      context_x = tf.gather(x_values, idx[:num_context], axis=1)
      context_y = tf.gather(y_values, idx[:num_context], axis=1)

    else:
      # Select the targets which will consist of the context points as well as
      # some new target points
      target_x = x_values[:, :num_target + num_context, :]
      target_y = y_values[:, :num_target + num_context, :]

      # Select the observations
      context_x = x_values[:, :num_context, :]
      context_y = y_values[:, :num_context, :]

    query = ((context_x, context_y), target_x)

    return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context)








# TIME SERIES CLASS
# data format should be as follows here:
# data = np.zeros((10000,4),dtype='float32')
# data[:,0] = [i//100 for i in range(10000)]
# data = tf.convert_to_tensor(data)
# dataset_train = PeriodicTSCurvesReader(16,4,data,100,testing=False)
# data_train = dataset_train.generate_curves()
# dataset_test = PeriodicTSCurvesReader(1,4,data,100,testing=True)
# data_test = dataset_test.generate_curves()
class PeriodicTSCurvesReader(object):
  """Generates curves from periodic time series data.
  """

  def __init__(self,
               batch_size,
               max_num_context,
               data,
               num_inst,
               testing=False):
    """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
    """
    self._batch_size = batch_size
    self._max_num_context = max_num_context
    self._data = data
    self._x_data = self._data[:,1:-1]
    self._y_data = self._data[:,-1]
    self._testing = testing
    self._num_inst = num_inst
    self._num_pts_per_inst = tf.cast(self._data.get_shape().as_list()[0]/self._num_inst,tf.int32)
    self._x_uniq = self._x_data[:self._num_pts_per_inst] #tf.unique(self._x_data)

  def generate_curves(self, seed=None):
    """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
    """
    num_context = tf.random_uniform(
        shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32, seed=seed)

    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.
    if self._testing:
      num_target = self._x_data.get_shape().as_list()[0]
      num_total_points = num_target
    # During training the number of target points and their x-positions are
    # selected at random
    else:
      num_target = tf.random_uniform(shape=(), minval=0, 
                                     maxval=self._max_num_context - num_context,
                                     dtype=tf.int32, seed=seed)
      num_total_points = num_context + num_target

    # idx for x vals in target
    idxs = []
    # which instance to get y data from
    insts = []
    for i in range(self._batch_size):
      idxs.append( tf.random_shuffle(tf.range(self._num_pts_per_inst), seed=seed) )
      insts.append( tf.random_uniform(shape=[], minval=0, maxval=self._num_inst-1, dtype=tf.int32, seed=seed) )
      
    idxs = tf.stack(idxs)
    insts = tf.stack(insts)
      
    # batchsize x numtotalpoints x size (xsize or ysize)
    x_values = tf.stack([tf.expand_dims(tf.gather(self._x_uniq, idxs[tf.cast(i,tf.int32)][:tf.cast(num_total_points,tf.int32)]), axis=-1) for i in range(self._batch_size)])
    y_values = tf.stack([tf.expand_dims(tf.gather(self._y_data[insts[i]*self._num_pts_per_inst:(insts[i]+1)*self._num_pts_per_inst], idxs[i][:num_total_points]), axis=-1) for i in range(self._batch_size)])
    
    
    
    if self._testing:
      # Select the targets
      target_x = x_values
      target_y = y_values

      # Select the observations
      idx_ctxt = tf.random_shuffle(tf.range(num_target), seed=seed)
      context_x = tf.gather(x_values, idx_ctxt[:num_context], axis=1)
      context_y = tf.gather(y_values, idx_ctxt[:num_context], axis=1)

    else:
      # Select the targets which will consist of the context points as well as
      # some new target points
      target_x = x_values[:, :num_target + num_context, :]
      target_y = y_values[:, :num_target + num_context, :]

      # Select the observations
      context_x = x_values[:, :num_context, :]
      context_y = y_values[:, :num_context, :]
      
    context_x = tf.squeeze(context_x,-1)
    target_x = tf.squeeze(target_x,-1)

    query = ((context_x, context_y), target_x)

    return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context)












class PeriodicNSCurvesReader(object):
  """Generates curves using a Gaussian Process (GP).

  Supports vector inputs (x) and vector outputs (y). Kernel is
  mean-squared exponential, using the x-value l2 coordinate distance scaled by
  some factor chosen randomly in a range. Outputs are independent gaussian
  processes.
  """

  def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               l1_scale=0.6,
               sigma_scale=1.0,
               epsilon=0.01,
               num_gammas=2,
               random_kernel_parameters=True,
               testing=False):
    """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      x_size: Integer >= 1 for length of "x values" vector.
      y_size: Integer >= 1 for length of "y values" vector.
      l1_scale: Float; typical scale for kernel distance function.
      sigma_scale: Float; typical scale for variance.
      random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma) 
          will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
    """
    self._batch_size = batch_size
    self._max_num_context = max_num_context
    self._x_size = x_size
    self._y_size = y_size
    self._l1_scale = l1_scale
    self._sigma_scale = sigma_scale
    self._random_kernel_parameters = random_kernel_parameters
    self._testing = testing
    self._epsilon = epsilon
    self._num_gammas = num_gammas

  def generate_curves(self, seed=None):
    """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
    """
    num_context = tf.random_uniform(
        shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32, seed=seed)

    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.
    if self._testing:
      num_target = 400
      num_total_points = num_target
      x_values = tf.tile(
          tf.expand_dims(tf.range(-2., 2., 1. / 100, dtype=tf.float32), axis=0),
          [self._batch_size, 1])
      x_values = tf.expand_dims(x_values, axis=-1)
    # During training the number of target points and their x-positions are
    # selected at random
    else:
      num_target = tf.random_uniform(shape=(), minval=0, 
                                     maxval=self._max_num_context - num_context,
                                     dtype=tf.int32, seed=seed)
      num_total_points = num_context + num_target
      x_values = tf.random_uniform(
          [self._batch_size, num_total_points, self._x_size], -2, 2, seed=seed)
    
    def w(x, x_min=-2, x_max=2):
      weight_vals = tf.stack([ [1/(i+1) if j <= i else 0 for j in range(self._num_gammas)] for i in range(self._num_gammas)])
      
      bucketsize = (x_max-x_min)/self._num_gammas
      buckets = (x-x_min)/bucketsize
      buckets = tf.reshape(buckets,[-1])
      
      mapped =  tf.expand_dims(tf.expand_dims(tf.map_fn(lambda x: weight_vals[tf.cast(x,tf.int32)], buckets),-2),-2)

      return mapped 

    # Set kernel parameters
    # Either choose a set of random parameters for the mini-batch
    if self._random_kernel_parameters:
      gammas = 3.14*tf.random_uniform([self._num_gammas, self._batch_size], 0.1, 2)
      gammas = tf.expand_dims(tf.expand_dims(gammas,-1),-1)
    # Or use the same fixed parameters for all mini-batches
    else:
      gammas = 3.14*tf.linspace(0.1,2,self._num_gammas)
      print(gammas)
      #gammas = tf.broadcast_to(gammas,[self._num_gammas, self._batch_size])
      gammas = tf.reshape(tf.tile(gammas,tf.constant([self._batch_size])),[self._num_gammas, self._batch_size])
      gammas = tf.expand_dims(tf.expand_dims(gammas,-1),-1)

    weights = w(x_values)
      
    weights = tf.reshape(weights, [self._batch_size, num_total_points,self._x_size,self._num_gammas])
    weights = tf.transpose(weights,[3,0,1,2])
    
    gammas = tf.broadcast_to(gammas,[self._num_gammas, self._batch_size, num_total_points, self._x_size])
    x_values_bcast = tf.expand_dims(x_values, 0)
    x_values_bcast = tf.broadcast_to(x_values_bcast,[self._num_gammas, self._batch_size, num_total_points, self._x_size])
    
    out = tf.math.multiply(gammas,x_values_bcast)
    out = tf.math.multiply(weights,tf.sin(out))
    out = tf.reduce_sum(out,axis=0)
    
    y_values = out
    y_values += tf.random.normal((self._batch_size,num_total_points,self._y_size),stddev = self._epsilon, seed=seed)

    if self._testing:
      # Select the targets
      target_x = x_values
      target_y = y_values

      # Select the observations
      idx = tf.random_shuffle(tf.range(num_target), seed=seed)
      context_x = tf.gather(x_values, idx[:num_context], axis=1)
      context_y = tf.gather(y_values, idx[:num_context], axis=1)

    else:
      # Select the targets which will consist of the context points as well as
      # some new target points
      target_x = x_values[:, :num_target + num_context, :]
      target_y = y_values[:, :num_target + num_context, :]

      # Select the observations
      context_x = x_values[:, :num_context, :]
      context_y = y_values[:, :num_context, :]

    query = ((context_x, context_y), target_x)

    return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context)












# IMAGE COMPLETION CLASS
# data format should be as follows here:
# imdata = np.zeros((100*32*32,6),dtype='float32')
# imdata[:,0] = [i//(32*32) for i in range(100*32*32)]
# imdata[:,1] = [(i/32)%32 for i in range(100*32*32)]
# imdata[:,2] = [i%32 for i in range(100*32*32)]
# imdata = tf.convert_to_tensor(imdata)
# dataset_train = ImageCompletionReader(16,3,10,imdata,100)
# data_train = dataset_train.generate_curves()
# dataset_test = ImageCompletionReader(1,3,10,imdata,100)
# data_test = dataset_train.generate_curves()
class ImageCompletionReader(object):
  """Generates curves from periodic time series data.
  """

  def __init__(self,
               batch_size,
               min_num_context,
               max_num_context,
               data,
               num_inst,
               testing=False):
    """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
    """
    self._batch_size = batch_size
    self._min_num_context = min_num_context
    self._max_num_context = max_num_context
    self._data = data
    # Hardcoded for right now
    self._x_data = self._data[:,1:-1]
    self._y_data = self._data[:,-1:]
    self._testing = testing
    self._num_inst = num_inst
    self._num_pts_per_inst = tf.cast(self._data.get_shape().as_list()[0]/self._num_inst,tf.int32)
    self._x_uniq = self._x_data[:self._num_pts_per_inst]

  def generate_curves(self, seed=None):
    """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
    """
    num_context = tf.random_uniform(
        shape=[], minval=self._min_num_context, maxval=self._max_num_context, dtype=tf.int32, seed=seed)

    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.
    if self._testing:
      num_target = self._num_pts_per_inst #self._x_data.get_shape().as_list()[0]
      num_total_points = num_target
    # During training the number of target points and their x-positions are
    # selected at random
    else:
      num_target = tf.random_uniform(shape=(), minval=0, 
                                     maxval=self._max_num_context - num_context,
                                     dtype=tf.int32, seed=seed)
      num_total_points = num_context + num_target

    # idx for x vals in target
    idxs = []
    # which instance to get y data from
    insts = []
    for i in range(self._batch_size):
      idxs.append( tf.random_shuffle(tf.range(self._num_pts_per_inst), seed=seed) )
      insts.append( tf.random_uniform(shape=[], minval=0, maxval=self._num_inst-1, dtype=tf.int32, seed=seed) )
      
    idxs = tf.stack(idxs)
    insts = tf.stack(insts)
      
    # batchsize x numtotalpoints x size (xsize or ysize)
    x_values = tf.stack([tf.expand_dims(tf.gather(self._x_uniq, idxs[tf.cast(i,tf.int32)][:tf.cast(num_total_points,tf.int32)]), axis=-1) for i in range(self._batch_size)])
    y_values = tf.stack([tf.expand_dims(tf.gather(self._y_data[insts[i]*self._num_pts_per_inst:(insts[i]+1)*self._num_pts_per_inst], idxs[i][:num_total_points]), axis=-1) for i in range(self._batch_size)])
    
    
    
    if self._testing:
      # Select the targets
      target_x = x_values
      target_y = y_values

      # Select the observations
      idx_ctxt = tf.random_shuffle(tf.range(num_target), seed=seed)
      context_x = tf.gather(x_values, idx_ctxt[:num_context], axis=1)
      context_y = tf.gather(y_values, idx_ctxt[:num_context], axis=1)

    else:
      # Select the targets which will consist of the context points as well as
      # some new target points
      target_x = x_values[:, :num_target + num_context, :]
      target_y = y_values[:, :num_target + num_context, :]

      # Select the observations
      context_x = x_values[:, :num_context, :]
      context_y = y_values[:, :num_context, :]
      
    context_x = tf.squeeze(context_x,-1)
    target_x = tf.squeeze(target_x,-1)

    context_y = tf.squeeze(context_y,-1)
    target_y= tf.squeeze(target_y,-1)

    query = ((context_x, context_y), target_x)

    return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context)
