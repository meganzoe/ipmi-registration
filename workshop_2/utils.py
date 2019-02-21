import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('default')
import math

def calcEntropies(A, B, num_bins = [32,32]):
  """
    function to calculate the joint and marginal entropies for two images

    INPUTS:    A: an image stored as a 2D matrix
               B: an image stored as a 2D matrix. B must the the same size as
                  A
               num_bins: a 2 element vector specifying the number of bins to
                   in the joint histogram for each image [default = 32, 32]

    OUTPUTS:   H_AB: the joint entropy between A and B
               H_A: the marginal entropy in A
               H_B the marginal entropy in B

    NOTE: if either of the images contain NaN values these pixels should be
    ignored when calculating the SSD.
  """
  #use histcounts2 function to generate joint histogram, an convert to
  #probabilities
  joint_hist,_,_ = np.histogram2d(A.flatten(), B.flatten(), bins = num_bins)
  probs_AB = joint_hist / np.sum(joint_hist)
  
  #calculate marginal probability distributions for A and B
  probs_A = np.sum(probs_AB, axis=1)
  probs_B = np.sum(probs_AB, axis=0)
    
  #calculate joint entropy and marginal entropies
  #note, when taking sums must check for nan values as
  #0 * log(0) = nan
  H_AB = -np.nansum(probs_AB * np.log(probs_AB))
  H_A = -np.nansum(probs_A* np.log(probs_A))
  H_B = -np.nansum(probs_B * np.log(probs_B))
  return H_AB, H_A, H_B

def calcNCC(A,B):
  """
  function to calculate the normalised cross correlation between
  two images
  
  INPUTS:    A: an image stored as a 2D matrix
             B: an image stored as a 2D matrix. B must the the same size as
                 A
  
  OUTPUTS:   NCC: the value of the normalised cross correlation
  
  NOTE: if either of the images contain NaN values these pixels should be
  ignored when calculating the SSD.

  """
  # use nanmean and nanstd functions to calculate mean and std dev of each
  # image
  mu_A = np.nanmean(A)
  mu_B = np.nanmean(B)
  sig_A = np.nanstd(A, ddof=1)
  sig_B = np.nanstd(B, ddof=1)
  # calculate NCC using nansum to ignore nan values when summing over pixels
  return np.nansum((A-mu_A)*(B-mu_B))/(A.size * sig_A * sig_B)


def calcSSD(A,B):
  """
  function to calculate the sum of squared differences between
  two images
  
  INPUTS:    A: an image stored as a 2D matrix
             B: an image stored as a 2D matrix. B must be the 
                same size as A
                
  OUTPUTS:   SSD: the value of the squared differences
  
  NOTE: if either of the images contain NaN values, these 
        pixels should be ignored when calculating the SSD.
  """
  # use nansum function to find sum of squared differences ignoring NaNs
  return np.nansum((A-B)*(A-B))


def dispImage(img, int_lims = [], ax = None):
  """
  function to display a grey-scale image that is stored in 'standard
  orientation' with y-axis on the 2nd dimension and 0 at the bottom

  INPUTS:   img: image to be displayed
            int_lims: the intensity limits to use when displaying the
               image, int_lims(1) = min intensity to display, int_lims(2)
               = max intensity to display [default min and max intensity
               of image]
            ax: if displaying an image on a subplot grid or on top of a
              second image, optionally supply the axis on which to display 
              the image.
  """

  #check if intensity limits have been provided, and if not set to min and
  #max of image
  if not int_lims:
    int_lims = [np.nanmin(img), np.nanmax(img)]
    #check if min and max are same (i.e. all values in img are equal)
    if int_lims[0] == int_lims[1]:
      #add one to int_lims(2) and subtract one from int_lims(1), so that
      #int_lims(2) is larger than int_lims(1) as required by imagesc
      #function
      int_lims[0] -= 1
      int_lims[1] += 1
  # take transpose of image to switch x and y dimensions and display with
  # first pixel having coordinates 0,0
  img = img.T
  if not ax:
    plt.imshow(img, cmap = 'gray', vmin = int_lims[0], vmax = int_lims[1], \
      origin='lower')
  else:
    ax.imshow(img, cmap = 'gray', vmin = int_lims[0], vmax = int_lims[1], \
      origin='lower')
  #set axis to be scaled equally (assumes isotropic pixel dimensions), tight
  #around the image
  plt.axis('image')
  plt.tight_layout()
  return ax

def defFieldFromAffineMatrix(aff_mat,num_pix_x, num_pix_y):
  """
  function to create a 2D deformation field from an affine matrix

  INPUTS:   aff_mat: a 3 x 3 numpy matrix representing the 2D affine 
                  transformation in homogeneous coordinates
           num_pix_x: number of pixels in the deformation field along the x
                  dimension
           num_pix_y: number of pixels in the deformation field along the y
                 dimension  

  OUTPUTS:  def_field: the 2D deformation field
  """
  # form 2S matrices containing all the pixel coordinates
  # note the rows/columns are switched for this meshgrid
  # function
  [Y,X] = np.meshgrid(range(num_pix_x), range(num_pix_y))
  # reshape and combine coordinate matrices into a 2 x N matrix, where N is
  # the total number of pixels (num_pix_x x num_pix_y)
  total_pix = num_pix_x * num_pix_y
  pix_coords = np.array([np.reshape(X,-1),np.reshape(Y,-1),np.ones(total_pix)])
  # apply transformation to pixel coordinates
  trans_coords = aff_mat * pix_coords
  #reshape into deformation field by first creating an empty deformation field
  def_field = np.zeros((num_pix_x, num_pix_y, 2))
  def_field[:,:,0] = np.reshape(trans_coords[0,:],(num_pix_x, num_pix_y))
  def_field[:,:,1] = np.reshape(trans_coords[1,:],(num_pix_x, num_pix_y))
  return def_field

def resampImageWithDefField(source_img, def_field, interp_method = 'linear'):
  """
  function to resample a 2D image with a 2D deformation field

  INPUTS:    source_img: the source image to be resampled, as a 2D matrix
             def_field: the deformation field, as a 3D matrix
             inter_method: any of the interpolation methods accepted by
                 interpn function [default = 'linear'] - 
                 'linear', 'nearest' and 'splinef2d'
  OUTPUTS:   resamp_img: the resampled image
  
  NOTES: the deformation field should be a 3D numpy array, where the size of the
  first two dimensions is the size of the resampled image, and the size of
  the 3rd dimension is 2. def_field[:,:,0] contains the x coordinates of the
  transformed pixels, def_field[:,:,1] contains the y coordinates of the
  transformed pixels.
  the origin of the source image is assumed to be the bottom left pixel
  """
  x_coords = np.arange(source_img.shape[0], dtype = 'float')
  y_coords = np.arange(source_img.shape[1], dtype = 'float')
  from scipy.interpolate import interpn
  # resample image using interpn function
  return interpn((x_coords,y_coords),source_img, def_field, bounds_error=False,\
    fill_value=np.NAN, method = interp_method)

def resampImageWithDefFieldPushInterp(source_img, def_field, interp_method = 'linear'):
  """
  function to resample a 2D image with a 2D deformation field using push
  interpolation
  
  INPUTS:    source_img: the source image to be resampled, as a 2D numpy matrix
                or numpy array
             def_field: the deformation field, as a 3D numpy array
             inter_method: 'linear' or 'nearest' [default = 'linear']
  OUTPUTS:   resamp_img: the resampled image
  
  NOTES: the deformation field should be a 3D numpy array where the size of the
  first two dimensions is the same as the source image, and the size of
  the 3rd dimension is 2. def_field[:,:,0] contains the x coordinates of the
  transformed pixels, def_field[:,:,1] contains the y coordinates of the
  transformed pixels.
  the resampled image will be the same size as the source image, and the
  origin is assumed to be the bottom left pixel
  """
  from scipy.interpolate import griddata
  x_coords = range(source_img.shape[0])
  y_coords = range(source_img.shape[1])
  #form matrices containing the pixel coordinates of the resmapled image
  [Y,X] = np.meshgrid(x_coords, y_coords)
  pix_coords = np.array([np.reshape(X,-1),np.reshape(Y,-1)]).T
  def_field_x = def_field[:,:,0]
  def_field_y = def_field[:,:,1]
  def_field_reformed = np.array([np.reshape(def_field_x,-1), np.reshape(def_field_y,-1)]).T
  # use scipy's griddata function to interpolate the irregular points in the
  # deformation field onto a regular grid
  resamp_img = griddata(def_field_reformed, np.reshape(source_img,-1), pix_coords, \
    method = interp_method)
  # reshape resampled image to have same size and shape as source image
  return np.reshape(resamp_img, source_img.shape)

def affineMatrixForRotationAboutPoint(theta, p_coords):
  """
  function to calculate the affine matrix corresponding to an anticlockwise
  rotation about a point
  
  INPUTS:    theta: the angle of the rotation, specified in degrees
             p_coords: the 2D coordinates of the point that is the centre of
                 rotation. p_coords[0] is the x coordinate, p_coords[1] is
                 the y coordinate
  
  OUTPUTS:   aff_mat: a 3 x 3 affine matrix
  """
  # convert theta to radians
  theta = math.pi * float(theta) / 180
  # form matrices for translations and rotation
  T1 = np.matrix([[1, 0, -p_coords[0]], [0, 1, -p_coords[1]], [0, 0, 1]])
  T2 = np.matrix([[1, 0, p_coords[0]], [0, 1, p_coords[1]], [0, 0, 1]])
  R = np.matrix([[math.cos(theta), -math.sin(theta), 0], \
      [math.sin(theta), math.cos(theta), 0], \
      [0, 0, 1]])

  # compose matrices
  return T2 * R * T1
