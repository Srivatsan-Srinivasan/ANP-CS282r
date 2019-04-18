# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:18:07 2019

@author: SrivatsanPC
"""
import matplotlib.pyplot as plt

#@title
def plot_1D_curves(target_x, target_y, context_x, context_y, pred_y, std, save = True, filename = 'default'):
  """Plots the predicted mean and variance and the context points.
  
  Args: 
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains 
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains 
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
  """
  # Plot everything
  plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
  plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
  plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
  plt.fill_between(
      target_x[0, :, 0],
      pred_y[0, :, 0] - std[0, :, 0],
      pred_y[0, :, 0] + std[0, :, 0],
      alpha=0.2,
      facecolor='#65c9f7',
      interpolate=True)

  # Make the plot pretty
  plt.yticks([-2, 0, 2], fontsize=16)
  plt.xticks([-2, 0, 2], fontsize=16)
  plt.ylim([-2, 2])
  plt.grid('off')
  ax = plt.gca()
  if not save :
      plt.show()
  else:
      plt.savefig(filename)

def plot_imgs(target_x,target_y,context_x,context_y,pred_y,std,save=True,filename = 'default',n_samp=5):
  nrows = max(target_x[:,:,0])+1
  ncols = max(target_x[:,:,1])+1

  pred_y = pred_y.reshape(nrows,ncols)
  std = std.reshape(nrows,ncols)

  mean = pred_y
  samps = np.random.normal(pred_y,std,size=(n_samp,pred_y.shape[0],pred_y.shape[1]))



  context_img = np.zeros((nrows,ncols))
  for i in range(len(context_x)):
    x = context_x[i]
    context_img[x[0],x[1]] = context_y[i]


  fig=plt.figure(figsize=(16, 16))
  columns = 1
  rows = n_samp+1
  for i in range(1, columns*rows +1):
      fig.add_subplot(rows,columns,i)
      if i == 1:
        plt.imshow(context_img)
      else:
        plt.imshow(samps[i-2])
  
  if not save:
      plt.show()
  else:
      plt.savefig(filename)

