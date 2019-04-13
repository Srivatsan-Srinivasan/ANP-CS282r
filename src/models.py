# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:26:57 2019

@author: SrivatsanPC
"""

from utils import batch_mlp
import tensorflow as tf


# TODO: add self-attention as an option
class LatentEncoder_cross(object):
  """The Latent Encoder."""

  def __init__(self, output_sizes, num_latents,attention,use_self_attention):
    """(A)NP latent encoder.

    Args:
      output_sizes: An iterable containing the output sizes of the encoding MLP.
      num_latents: The latent dimensionality.
    """
    self._output_sizes = output_sizes
    print("Using cross attention in the latent encoder ")
    self._num_latents = num_latents
    self._attention = attention
    
    self._use_self_attention = use_self_attention

  def __call__(self, x_cont, y_cont, x_targ):
    """Encodes the inputs into one representation.

    Args:
      x: Tensor of shape [B,observations,d_x]. For this 1D regression
          task this corresponds to the x-values.
      y: Tensor of shape [B,observations,d_y]. For this 1D regression
          task this corresponds to the y-values.

    Returns:
      A normal distribution over tensors of shape [B, num_latents]
    """

    # Concatenate x and y along the filter axes
    encoder_input = tf.concat([x_cont, y_cont], axis=-1)
    
    if self._use_self_attention:
      print('Using self attention in the latent cross encoder')
      encoder_input = batch_mlp(encoder_input, [2,128,128],"latent_encoder_cross_self")  

      with tf.variable_scope("latent_encoder_cross_self", reuse=tf.AUTO_REUSE):
        encoder_input = self._attention(encoder_input,encoder_input,encoder_input)
    
    # Pass final axis through MLP
    hidden_mean = batch_mlp(encoder_input, [2,self._num_latents], "latent_encode_cross_mean")
    hidden_var = batch_mlp(encoder_input, [2,self._num_latents], "latent_encode_cross_var")
    
    sigma = 0.1 + 0.9 * tf.sigmoid(hidden_var)
    
    z_samps = tf.contrib.distributions.Normal(loc = hidden_mean, scale = sigma).sample()
    
    with tf.variable_scope("latent_cross", reuse=tf.AUTO_REUSE):        
        z_star = self._attention(x_cont,x_targ,z_samps)
    
    return z_star, z_samps

class LatentEncoder(object):
  """The Latent Encoder."""

  def __init__(self, output_sizes, num_latents, attention, use_self_attention):
    """(A)NP latent encoder.

    Args:
      output_sizes: An iterable containing the output sizes of the encoding MLP.
      num_latents: The latent dimensionality.
    """
    self._output_sizes = output_sizes
    self._num_latents = num_latents
    self._attention = attention
    
    self._use_self_attention = use_self_attention

  def __call__(self, x, y):
    """Encodes the inputs into one representation.

    Args:
      x: Tensor of shape [B,observations,d_x]. For this 1D regression
          task this corresponds to the x-values.
      y: Tensor of shape [B,observations,d_y]. For this 1D regression
          task this corresponds to the y-values.

    Returns:
      A normal distribution over tensors of shape [B, num_latents]
    """

    # Concatenate x and y along the filter axes
    encoder_input = tf.concat([x, y], axis=-1)
    
    if self._use_self_attention:
      print('Using self attention in the latent encoder')
      encoder_input = batch_mlp(encoder_input, [2,128,128],"latent_encoder_self")  

      with tf.variable_scope("latent_encoder_self", reuse=tf.AUTO_REUSE):
        encoder_input = self._attention(encoder_input,encoder_input,encoder_input)
    
    # Pass final axis through MLP
    hidden = batch_mlp(encoder_input, self._output_sizes, "latent_encoder")
    
    # Aggregator: take the mean over all points
    hidden = tf.reduce_mean(hidden, axis=1)
    
    # Have further MLP layers that map to the parameters of the Gaussian latent
    with tf.variable_scope("latent_encoder", reuse=tf.AUTO_REUSE):
      # First apply intermediate relu layer 
      hidden = tf.nn.relu(
          tf.layers.dense(hidden, 
                          (self._output_sizes[-1] + self._num_latents)/2, 
                          name="penultimate_layer"))
      # Then apply further linear layers to output latent mu and log sigma
      mu = tf.layers.dense(hidden, self._num_latents, name="mean_layer")
      log_sigma = tf.layers.dense(hidden, self._num_latents, name="std_layer")
      
    # Compute sigma
    sigma = 0.1 + 0.9 * tf.sigmoid(log_sigma)

    return tf.contrib.distributions.Normal(loc=mu, scale=sigma)


class DeterministicEncoder(object):
  """The Deterministic Encoder."""

  def __init__(self, output_sizes, attention, use_self_attention = False):
    """(A)NP deterministic encoder.

    Args:
      output_sizes: An iterable containing the output sizes of the encoding MLP.
      attention: The attention module.
    """
    self._output_sizes = output_sizes
    self._attention = attention
    
    self._use_self_attention = use_self_attention

  def __call__(self, context_x, context_y, target_x):
    """Encodes the inputs into one representation.

    Args:
      context_x: Tensor of shape [B,observations,d_x]. For this 1D regression
          task this corresponds to the x-values.
      context_y: Tensor of shape [B,observations,d_y]. For this 1D regression
          task this corresponds to the y-values.
      target_x: Tensor of shape [B,target_observations,d_x]. 
          For this 1D regression task this corresponds to the x-values.

    Returns:
      The encoded representation. Tensor of shape [B,target_observations,d]
    """

    # Concatenate x and y along the filter axes
    encoder_input = tf.concat([context_x, context_y], axis=-1)
    
    if self._use_self_attention:
      print('Uaing self attention in the deterministic encoder')
      encoder_input = batch_mlp(encoder_input, [2,128,128],"deterministic_encoder_self")  

      with tf.variable_scope("deterministic_encoder_self", reuse=tf.AUTO_REUSE):
        encoder_input = self._attention(encoder_input,encoder_input,encoder_input)
        
    # Pass final axis through MLP
    hidden = batch_mlp(encoder_input, self._output_sizes, 
                       "deterministic_encoder")

    # Apply attention
    with tf.variable_scope("deterministic_encoder", reuse=tf.AUTO_REUSE):
        hidden = self._attention(context_x, target_x, hidden)

    return hidden


class Decoder(object):
  """The Decoder."""

  def __init__(self, output_sizes, apply_attention=True, attention = None):
    """(A)NP decoder.

    Args:
      output_sizes: An iterable containing the output sizes of the decoder MLP 
          as defined in `basic.Linear`.
    """
    self._output_sizes = output_sizes
   
    self._apply_attention = apply_attention   
    
    if self._apply_attention:
      self._attention = attention
          
    print('Decoder Attention is ', self._apply_attention)
    
    

  def __call__(self, representation, target_x):
    """Decodes the individual targets.

    Args:
      representation: The representation of the context for target predictions. 
          Tensor of shape [B,target_observations,?].
      target_x: The x locations for the target query.
          Tensor of shape [B,target_observations,d_x].

    Returns:
      dist: A multivariate Gaussian over the target points. A distribution over
          tensors of shape [B,target_observations,d_y].
      mu: The mean of the multivariate Gaussian.
          Tensor of shape [B,target_observations,d_x].
      sigma: The standard deviation of the multivariate Gaussian.
          Tensor of shape [B,target_observations,d_x].
    """
   
    
    if self._apply_attention:
      print('Using self attention in the decoder')
      hidden_decoder = batch_mlp(target_x, [1,128,128],"hidden_decoder")    
      print("Input Hidden ", hidden_decoder)
      with tf.variable_scope("hidden_decoder", reuse=tf.AUTO_REUSE):        
        hidden_decoder = self._attention(hidden_decoder ,hidden_decoder,hidden_decoder)
      print("Attention Hidden ", hidden_decoder)     
    
      hidden = tf.concat([representation, hidden_decoder], axis=-1)
    else:
      hidden = tf.concat([representation, target_x], axis=-1)
      
    
    # Pass final axis through MLP
    hidden = batch_mlp(hidden, self._output_sizes, "decoder")
    print("Output hidden ", hidden)
   
   
    # Get the mean an the variance
    mu, log_sigma = tf.split(hidden, 2, axis=-1)

    # Bound the variance
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

    # Get the distribution
    dist = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)

    return dist, mu, sigma

class LatentModel(object):
  """The (A)NP model."""

  def __init__(self, latent_encoder_output_sizes, num_latents,
               decoder_output_sizes, use_deterministic_path=True, 
               deterministic_encoder_output_sizes=None, attention=None, 
               use_attention_decoder=False,
               use_encoder_determ_self_attention = False,
               use_encoder_latent_self_attention = False,
               use_encoder_latent_cross_attention = False):
    """Initialises the model.

    Args:
      latent_encoder_output_sizes: An iterable containing the sizes of hidden 
          layers of the latent encoder.
      num_latents: The latent dimensionality.
      decoder_output_sizes: An iterable containing the sizes of hidden layers of
          the decoder. The last element should correspond to d_y * 2
          (it encodes both mean and variance concatenated)
      use_deterministic_path: a boolean that indicates whether the deterministic
          encoder is used or not.
      deterministic_encoder_output_sizes: An iterable containing the sizes of 
          hidden layers of the deterministic encoder. The last one is the size 
          of the deterministic representation r.
      attention: The attention module used in the deterministic encoder.
          Only relevant when use_deterministic_path=True.
    """
    self._latent_encoder = LatentEncoder(latent_encoder_output_sizes, 
                                         num_latents,attention, use_encoder_latent_self_attention)
    self._use_encoder_latent_cross_attention = use_encoder_latent_cross_attention
    if use_encoder_latent_cross_attention:
      self._latent_encoder_cross = LatentEncoder_cross(latent_encoder_output_sizes,
                                                    num_latents,attention,use_encoder_latent_self_attention)
    self._decoder = Decoder(decoder_output_sizes, apply_attention = use_attention_decoder, attention = attention)
    self._use_deterministic_path = use_deterministic_path
    if use_deterministic_path:
      self._deterministic_encoder = DeterministicEncoder(
          deterministic_encoder_output_sizes, attention, use_encoder_determ_self_attention)
    
  def __call__(self, query, num_targets, target_y=None):
    """Returns the predicted mean and variance at the target points.

    Args:
      query: Array containing ((context_x, context_y), target_x) where:
          context_x: Tensor of shape [B,num_contexts,d_x]. 
              Contains the x values of the context points.
          context_y: Tensor of shape [B,num_contexts,d_y]. 
              Contains the y values of the context points.
          target_x: Tensor of shape [B,num_targets,d_x]. 
              Contains the x values of the target points.
      num_targets: Number of target points.
      target_y: The ground truth y values of the target y. 
          Tensor of shape [B,num_targets,d_y].

    Returns:
      log_p: The log_probability of the target_y given the predicted
          distribution. Tensor of shape [B,num_targets].
      mu: The mean of the predicted distribution. 
          Tensor of shape [B,num_targets,d_y].
      sigma: The variance of the predicted distribution.
          Tensor of shape [B,num_targets,d_y].
    """

    (context_x, context_y), target_x = query

    # Pass query through the encoder and the decoder
    prior = self._latent_encoder(context_x, context_y)
    if self._use_encoder_latent_cross_attention:
      latent_cross,z_samps_prior = self._latent_encoder_cross(context_x, context_y, target_x)
    
    # For training, when target_y is available, use targets for latent encoder.
    # Note that targets contain contexts by design.
    if target_y is None:
      latent_rep = prior.sample()
    # For testing, when target_y unavailable, use contexts for latent encoder.
    else:
      posterior = self._latent_encoder(target_x, target_y)
      latent_rep = posterior.sample()
      if self._use_encoder_latent_cross_attention:
        latent_cross,z_samps_posterior = self._latent_encoder_cross(target_x, target_y, target_x) 
    
    latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                         [1, num_targets, 1])
        
    if self._use_deterministic_path:
      deterministic_rep = self._deterministic_encoder(context_x, context_y,
                                                      target_x)

      if self._use_encoder_latent_cross_attention:    
        representation = tf.concat([deterministic_rep, latent_cross, latent_rep], axis=-1)
      else:
        representation = tf.concat([deterministic_rep,latent_rep],axis=-1)
        
    else:
      if self._use_encoder_latent_cross_attention:
        representation = tf.concat([latent_cross, latent_rep], axis=-1)
      else:
        representation = latent_rep
      
    dist, mu, sigma = self._decoder(representation, target_x)
    
    # If we want to calculate the log_prob for training we will make use of the
    # target_y. At test time the target_y is not available so we return None.
    if target_y is not None:
      log_p = dist.log_prob(target_y)
      posterior = self._latent_encoder(target_x, target_y)
      
      kl = tf.reduce_sum(
          tf.contrib.distributions.kl_divergence(posterior, prior), 
          axis=-1, keepdims=True)
      kl = tf.tile(kl, [1, num_targets])
      
#       # TODO: KL for cross_attention part of latent encoder
#       kl_cross = tf.reduce_sum(
#           tf.contrib.distributions.kl_divergence(z_samps_posterior, z_samps_prior), 
#           axis=-1, keepdims=True)
#       kl_cross = tf.tile(kl_cross, [1, num_targets])
      
      loss = - tf.reduce_mean(log_p - kl / tf.cast(num_targets, tf.float32))
    else:
      log_p = None
      kl = None
      loss = None

    return mu, sigma, log_p, kl, loss
