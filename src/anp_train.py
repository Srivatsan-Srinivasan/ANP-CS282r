# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:19:29 2019

@author: SrivatsanPC
"""
import tensorflow as tf
import numpy as np
from curveReader import GPCurvesReader
from plotting import plot_1D_curves, plot_imgs
from attention import Attention
from models import LatentModel
from utils import get_data, get_errors_1D
import os

def train_anp(args):
    MODEL_SAVENAME = args.model_savename
    TRAINING_ITERATIONS = args.n_iter 
    MAX_CONTEXT_POINTS = args.n_context_max 
    MIN_CONTEXT_POINTS = args.n_context_min
    PLOT_AFTER = args.plot_after
    LOSS_AFTER = args.loss_after
    HIDDEN_SIZE = args.h_size 
    MODEL_TYPE = args.model_type

    #hardcode uniform attention for simple NP - returns mean essentially.
    if MODEL_TYPE != 'NP':
        ATTENTION_TYPE = args.attention 
    else:
        ATTENTION_TYPE = 'uniform'

    KERNEL = args.kernel
    DATA_FORMAT = args.data_format
    decoder_output_size = args.decoder_output_size
    num_gammas = args.num_gammas
    
    
    random_kernel_parameters= args.random_kernel_params 
    use_decoder_self_attention = args.SA_decoder 
    use_encoder_determ_self_attention = args.SA_det_encoder
    use_encoder_latent_self_attention = args.SA_lat_encoder 
    use_encoder_latent_cross_attention = args.CA_lat_encoder 

    #Printing key arguments for sanity
    print('Key argument : Model Type : {}, Attention : {}, Kernel :{}, Data Format :{}, decoder self :{}, encoder self : {}, latent self :{}, latent cross :{}'.format(MODEL_TYPE, ATTENTION_TYPE, KERNEL, DATA_FORMAT, use_decoder_self_attention, use_encoder_determ_self_attention, use_encoder_latent_self_attention, use_encoder_latent_cross_attention)) 

    if DATA_FORMAT == 'GP':
        modelname = MODEL_TYPE + '_'+  KERNEL
    else:
        modelname = MODEL_TYPE

    filename = '{}_{}_{}_{}_{}_{}_{}'.format(DATA_FORMAT, modelname, ATTENTION_TYPE,
                                                use_decoder_self_attention,
                                                use_encoder_determ_self_attention,
                                                use_encoder_latent_self_attention,
                                                use_encoder_latent_cross_attention)    
    
    tf.reset_default_graph()
        
    data_train, data_test = get_data(DATA_FORMAT, kernel=KERNEL, max_context_points = MAX_CONTEXT_POINTS,
                                     random_kernel_parameters = random_kernel_parameters, 
                                     train_batch_size = args.train_batch_size, test_batch_size = args.test_batch_size, min_context_points = MIN_CONTEXT_POINTS, num_gammas = num_gammas)

    print('Data Generated!')
    
    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    latent_encoder_output_sizes = [HIDDEN_SIZE]*4
    num_latents = HIDDEN_SIZE
    deterministic_encoder_output_sizes= [HIDDEN_SIZE]*4
    decoder_output_sizes = [HIDDEN_SIZE]*2 + [decoder_output_size*2]
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
    loss_arr = np.zeros(TRAINING_ITERATIONS//LOSS_AFTER)
    context_mse_arr, context_nll_arr, target_mse_arr, target_nll_arr = np.zeros(TRAINING_ITERATIONS//LOSS_AFTER), np.zeros(TRAINING_ITERATIONS//LOSS_AFTER), np.zeros(TRAINING_ITERATIONS//LOSS_AFTER), np.zeros(TRAINING_ITERATIONS//LOSS_AFTER)


    # Train and plot
    with tf.train.MonitoredSession() as sess:
        sess.run(init)    
        for it in range(TRAINING_ITERATIONS):
            sess.run([train_step])    
            
            if it % LOSS_AFTER == 0:
                loss_value, pred_y, std_y, target_y, whole_query = sess.run(
                  [loss, mu, sigma, data_test.target_y, 
                   data_test.query]) 
                loss_arr[it//LOSS_AFTER] = loss_value                

                (context_x, context_y), target_x = whole_query
                context_mse_arr[it//LOSS_AFTER], context_nll_arr[it//LOSS_AFTER], target_mse_arr[it//LOSS_AFTER], target_nll_arr[it//LOSS_AFTER] = get_errors_1D(context_x, context_y, target_x, target_y, pred_y, std_y)

                loss_filename = os.path.join('loss_collection', 'loss_'+ filename + '.npy')
                con_nll_filename = os.path.join('loss_collection', 'context_nll_'+ filename + '.npy')
                con_mse_filename = os.path.join('loss_collection', 'context_mse_'+ filename + '.npy')
                tar_nll_filename = os.path.join('loss_collection', 'target_nll_' + filename + '.npy')
                tar_mse_filename = os.path.join('loss_collection', 'tar_mse_' + filename + '.npy')

                np.save(loss_filename,loss_arr)
                np.save(con_nll_filename, context_nll_arr)
                np.save(tar_nll_filename, target_nll_arr)
                np.save(tar_mse_filename, target_mse_arr)
                np.save(con_mse_filename, context_mse_arr)
                
            # Plot the predictions in `PLOT_AFTER` intervals    
            if it % PLOT_AFTER == 0:                
                (context_x, context_y), target_x = whole_query
                print('Iteration: {}, loss: {}'.format(it, loss_value))    
                # Plot the prediction and the context
                
                curves_filename = 'curves_'+ filename + '_' + str(it)  
                fname = os.path.join('temp_img',curves_filename + '.png')
                if DATA_FORMAT in ['GP','TS','per_NS']:
                    plot_1D_curves(target_x, target_y, context_x, context_y, pred_y, std_y,filename = fname)
                elif DATA_FORMAT in ['mnist']:
                    plot_imgs(target_x, target_y, context_x, context_y, pred_y, std_y,filename = fname)
    
        saver = tf.train.Saver()
        save_path = "training_{}/{}.ckpt".format(DATA_FORMAT,MODEL_SAVENAME)
        saving_to_path = saver.save(sess, save_path)
        print("Model saved in path: %s" % saving_to_path)

        # saver = tf.train.Saver()
        # # Restore variables from disk.
        # saver.restore(sess, "/tmp/model.ckpt")
        # print("Model restored.")



