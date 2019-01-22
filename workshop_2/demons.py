from skimage.transform import rescale
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import utils 
from matplotlib.colors import ListedColormap

class DemonsAlgorithm(object):
  def __init__(self, optDict = {}):
    # set the parameters of the algorithm
    # if the parameters have not been set in the options dictionary
    # then use the default parameters defined in the second
    # argument of dictionary.get()
    self.target_full = optDict.get('target', '')
    self.source_full = optDict.get('source','')
    self.sigma_elastic = optDict.get('sigma_elastic', 1)
    self.sigma_fluid = optDict.get('sigma_fluid', 1)
    self.use_target_grad = optDict.get('use_target_grad', False)
    self.num_lev = optDict.get('num_lev', 1)
    self.max_it = optDict.get('max_it', 100)
    self.df_thresh = optDict.get('df_thresh', 0.001)
    self.check_MSD = optDict.get('check_MSD', True)
    self.disp_freq = optDict.get('disp_freq', 10)

  def run( self ):
    # loop over resolution levels
    for lev in xrange(1,self.num_lev+1):
        # resample images if needed
        if lev == self.num_lev:
            target = self.target_full
            source = self.source_full
        else:
            resamp_factor = np.power(2, self.num_lev - lev)
            target = rescale(self.target_full, 1.0/resamp_factor, mode='reflect', preserve_range=True, \
                             multichannel=False, order =3,  anti_aliasing=True)
            source = rescale(self.source_full, 1.0/resamp_factor, mode='reflect', preserve_range=True, \
                             multichannel=False, order =3,  anti_aliasing=True)
            
        # if first level initialise def_field, disp_field and update
        if lev == 1:
            X,Y = np.mgrid[0:target.shape[0],0:target.shape[1]]
            def_field = np.zeros((X.shape[0],X.shape[1],2))
            def_field[:,:,0] = X
            def_field[:,:,1] = Y
            disp_field_x = np.zeros(target.shape)
            disp_field_y = np.zeros(target.shape)
            update_x = np.zeros(target.shape)
            update_y = np.zeros(target.shape)
        else:
            # otherwise upsample disp_field from previous level
            disp_field_x = 2 * rescale(disp_field_x,2, mode='reflect', preserve_range=True, \
                                       multichannel=False,order =3)
            disp_field_y = 2 * rescale(disp_field_y,2, mode='reflect', preserve_range=True, \
                                       multichannel=False,order =3)
            # recalculate def_field for this level from disp_field
            X,Y = np.mgrid[0:target.shape[0],0:target.shape[1]]
            def_field = np.zeros((X.shape[0],X.shape[1],2)) # clear def_field from previous level
            def_field[:,:,0] = X
            def_field[:,:,1] = Y
            update_x = np.zeros(target.shape)
            update_y = np.zeros(target.shape)
        
        # calculate the transformed image at the start of this level
        self.trans_image = utils.resampImageWithDefField(source, def_field)
        
        # store the current def_field and MSD value to check for improvements at 
        # end of iteration 
        def_field_prev = def_field.copy()
        prev_MSD = self._calcMSD(target, self.trans_image)
        
        # pre-calculate the image gradients. only one of source or target
        # gradients needs to be calculated, as indicated by use_target_grad
        if self.use_target_grad:
            [target_gradient_x, target_gradient_y] = np.gradient(target)
        else:
            [source_gradient_x, source_gradient_y] = np.gradient(source)
        
        ## DISPLAY RESULTS ##
        cmap1 = plt.cm.Reds_r
        my_cmap1 = cmap1(np.arange(cmap1.N))
        my_cmap1[:,-1] = np.linspace(0, 1, cmap1.N)
        my_cmap1 = ListedColormap(my_cmap1)
        cmap2 = plt.cm.Blues_r
        my_cmap2 = cmap2(np.arange(cmap2.N))
        my_cmap2[:,-1] = np.linspace(0, 1, cmap2.N)
        my_cmap2 = ListedColormap(my_cmap2)

        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (7,7))
        utils.dispImage(source, ax = ax[0,0])
        ax[0,0].set_title('Source lv: {}'.format(lev))
        utils.dispImage(target, ax = ax[0,1])
        ax[0,1].set_title('Target lv: {}'.format(lev))
        utils.dispImage(self.trans_image, ax = ax[1,0])
        ax[1,0].set_title('Transformed Image lv: {}'.format(lev))
        #utils.dispImage(target - self.trans_image, ax = ax[1,1])
        ax[1,1].imshow(target.T, cmap = my_cmap1, vmin = np.min(target), vmax = np.max(target), \
          origin='lower', alpha = 0.5)
        ax[1,1].imshow(self.trans_image.T, cmap = my_cmap2, vmin = np.min(target), vmax = np.max(target), \
          origin='lower', alpha = 0.5)
        ax[1,1].set_title('Target (red) & Transformed Image (blue)')
        #plt.show()
        
        # main iterative loop - repeat until max number of iterations reached
        for it in range(self.max_it):
            # calculate update from demons forces
            #
            # if using target image graident use as is
            if self.use_target_grad:
                img_grad_x = target_gradient_x
                img_grad_y = target_gradient_y
            else:
                # but if using source image gradient need to transform with
                # current deformation field
                img_grad_x = utils.resampImageWithDefField(source_gradient_x, def_field)
                img_grad_y = utils.resampImageWithDefField(source_gradient_y, def_field)
                
            # calculate difference image
            diff = target - self.trans_image
            #calculate denominator of demons forces
            denom = np.power(img_grad_x,2) + np.power(img_grad_y,2) + np.power(diff,2)
            #calculate x and y components of numerator of demons forces
            numer_x = diff * img_grad_x
            numer_y = diff * img_grad_y
            #calculate the x and y components of the update
            update_x = numer_x / denom
            update_y = numer_y / denom
            #set nan values to 0
            update_x[np.isnan(update_x)] = 0
            update_y[np.isnan(update_y)] = 0
            
            #if fluid like regularisation used smooth the update
            if self.sigma_fluid > 0:
                update_x = gaussian_filter(update_x, self.sigma_fluid, mode='nearest')
                update_y = gaussian_filter(update_y, self.sigma_fluid, mode='nearest')
            
            #add the update to the current displacement field
            disp_field_x = disp_field_x + update_x
            disp_field_y = disp_field_y + update_y
            
            #if elastic like regularisation used smooth the displacement field
            if self.sigma_elastic > 0:
                disp_field_x = gaussian_filter(disp_field_x,self.sigma_elastic, mode='nearest')
                disp_field_y = gaussian_filter(disp_field_y, self.sigma_elastic, mode='nearest')
            
            #update deformation field from disp field
            def_field[:,:,0] = disp_field_x + X
            def_field[:,:,1] = disp_field_y + Y
            
            #transform the image using the updated deformation field
            self.trans_image = utils.resampImageWithDefField(source, def_field)
            if it % self.disp_freq == 0:
              # plot the results
              utils.dispImage(self.trans_image, ax = ax[1,0])
              #utils.dispImage(target - self.trans_image, ax = ax[1,1])
              ax[1,1].imshow(target.T, cmap = my_cmap1, vmin = np.min(target), vmax = np.max(target), \
               origin='lower', alpha = 0.5)
              ax[1,1].imshow(self.trans_image.T, cmap = my_cmap2, vmin = np.min(target), vmax = np.max(target), \
                origin='lower', alpha = 0.5)
              fig.canvas.draw()

              fig2, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (7.5,5))
              utils.dispImage(disp_field_x, ax = axs[0,0])
              axs[0,0].set_title('Disp field x it: {}'.format(it))
              utils.dispImage(disp_field_y, ax = axs[1,0])
              axs[1,0].set_title('Disp field y it: {}'.format(it))
              utils.dispImage(update_x, ax = axs[0,1])
              axs[0,1].set_title('Update x it: {}'.format(it))
              utils.dispImage(update_y, ax = axs[1,1])
              axs[1,1].set_title('Update y it: {}'.format(it))
              utils.dispImage(self.trans_image, ax = axs[0,2])
              axs[0,2].set_title('Transformed image it: {}'.format(it))
              utils.dispImage(target - self.trans_image, ax = axs[1,2])
              axs[1,2].set_title('Target - transformed image it: {}'.format(it))
            
            
            #calculate MSD between target and transformed image
            MSD =  self._calcMSD(target, self.trans_image)

            #calculate max difference between previous and current def_field 
            max_df = np.max(np.abs(def_field - def_field_prev))

            #display numerical results
            if it % 5 == 0:
              print('Iteration {0:d}: MSD = {1:e}, Max. change in def field = {2:0.3f}\n'.format(it, MSD, max_df))
            
            #check if max change in def field below threshhold
            if max_df < self.df_thresh:
                print('max df < df_thresh')
                break
            
            #check for improvement in MSD if required
            if self.check_MSD and MSD > prev_MSD:
                #restore previous results and finish level
                def_field = def_field_prev
                MSD = prev_MSD
                self.trans_image = utils.resampImageWithDefField(source, def_field)
                print('No improvement in MSD')
                break
            
            #update previous values of def_field and MSD
            def_field_prev = def_field.copy()
            prev_MSD = MSD.copy()


  def _calcMSD(self, A, B):
    """
    function to calculate the  mean of squared differences between
    two images
    
    INPUTS:    A: an image stored as a 2D matrix
               B: an image stored as a 2D matrix. B must be the 
                  same size as A
                  
    OUTPUTS:   MSD: the value of the mean of squared differences
    
    
    NOTE: if either of the images contain NaN values, these 
          pixels should be ignored when calculating the MSD.
    """
    # use nansum function to find mean of squared differences ignoring NaNs
    return np.nanmean((A-B)*(A-B))