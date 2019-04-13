# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:19:29 2019

@author: SrivatsanPC
"""
import tensorflow as tf
import numpy as np
from curveReader import GPCurvesReader
from plotting import plot_1D_curves
from attention import Attention
from models import LatentModel
import os

def train_anp(args):
    TRAINING_ITERATIONS = args.n_iter 
    MAX_CONTEXT_POINTS = args.n_context_max 
    PLOT_AFTER = args.plot_after
    LOSS_AFTER = args.loss_after
    HIDDEN_SIZE = args.h_size 
    MODEL_TYPE = args.model_type
    ATTENTION_TYPE = args.attention #@param 
    KERNEL = args.kernel
    
    
    random_kernel_parameters= args.random_kernel_params 
    use_decoder_self_attention = args.SA_decoder 
    use_encoder_determ_self_attention = args.SA_det_encoder
    use_encoder_latent_self_attention = args.SA_lat_encoder 
    use_encoder_latent_cross_attention = args.CA_lat_encoder 
    
    filename = 'anp_loss_arr_{}_{}_{}_{}_{}_{}'.format(KERNEL,ATTENTION_TYPE,
                                                use_decoder_self_attention,
                                                use_encoder_determ_self_attention,
                                                use_encoder_latent_self_attention,
                                                use_encoder_latent_cross_attention)    
    
    tf.reset_default_graph()
    # Train dataset
    dataset_train = GPCurvesReader(
        batch_size=16, max_num_context=MAX_CONTEXT_POINTS, 
        random_kernel_parameters=random_kernel_parameters, kernel = KERNEL)
    data_train = dataset_train.generate_curves()
    
    # Test dataset
    dataset_test = GPCurvesReader(
        batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True,
        random_kernel_parameters=random_kernel_parameters, kernel = KERNEL)
    
    data_test = dataset_test.generate_curves()
    
    
    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    latent_encoder_output_sizes = [HIDDEN_SIZE]*4
    num_latents = HIDDEN_SIZE
    deterministic_encoder_output_sizes= [HIDDEN_SIZE]*4
    decoder_output_sizes = [HIDDEN_SIZE]*2 + [2]
    use_deterministic_path = True
    
    # ANP with multihead attention
    if MODEL_TYPE == 'ANP':
      attention = Attention(rep='mlp', output_sizes=[HIDDEN_SIZE]*2, 
                            att_type = ATTENTION_TYPE)
    # NP - equivalent to uniform attention
    elif MODEL_TYPE == 'NP':
      attention = Attention(rep='identity', output_sizes=None, att_type='uniform')
    else:
      raise NameError("MODEL_TYPE not among ['ANP,'NP']")
    
    # Define the model
    model = LatentModel(latent_encoder_output_sizes, num_latents,
                        decoder_output_sizes, use_deterministic_path, 
                        deterministic_encoder_output_sizes, attention, 
                        use_attention_decoder = use_decoder_self_attention,
                        use_encoder_determ_self_attention = use_encoder_determ_self_attention,
                        use_encoder_latent_self_attention = use_encoder_latent_self_attention,
                        use_encoder_latent_cross_attention = use_encoder_latent_cross_attention)
    
    # Define the loss
    _, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
                                     data_train.target_y)
    
    # Get the predicted mean and variance at the target points for the testing set
    mu, sigma, _, _, _ = model(data_test.query, data_test.num_total_points)
    
    # Set up the optimizer and train step
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(loss)
    init = tf.initialize_all_variables()
    
    # set up loss array
    loss_arr = np.zeros(TRAINING_ITERATIONS)
    
    # Train and plot
    with tf.train.MonitoredSession() as sess:
        sess.run(init)    
        for it in range(TRAINING_ITERATIONS):
            sess.run([train_step])    
            
            if it % LOSS_AFTER == 0:
                loss_value, pred_y, std_y, target_y, whole_query = sess.run(
                  [loss, mu, sigma, data_test.target_y, 
                   data_test.query]) 
                loss_arr[it] = loss_value   

                loss_filename = os.path.join('loss_collection', filename + '.npy')
                np.save(loss_filename,loss_arr)
                
            # Plot the predictions in `PLOT_AFTER` intervals    
            if it % PLOT_AFTER == 0:                
                (context_x, context_y), target_x = whole_query
                print('Iteration: {}, loss: {}'.format(it, loss_value))    
                # Plot the prediction and the context
                
                curves_filename = 'curves_'+ filename + '_' + str(it)  
                fname = os.path.join('temp_img',curves_filename + '.png')
                plot_1D_curves(target_x, target_y, context_x, context_y, pred_y, std_y,filename = fname)
          
        
          
        
          