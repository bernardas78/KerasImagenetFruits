# Performs color PCA:
#   Find means of R,G,B values over the entire train set
#   Calculate covariance matrix between colors [3;3]
#   Find eigenvectors (new coordinate axis) and eigenvalues (signigficance of variance over the axis); [3;3] and [3]
#   Eigenvectors, -values used to distort images while training
#          saved in 
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
#   exec(open("PCA.py").read())

from DataGen import AugSequence_v3_randomcrops as as_v3
import numpy as np
from numpy.linalg import eig
from PIL import Image

crop_range = 1 
target_size = 255
datasrc = "ilsvrc14_50classes"

dataGen = as_v3.AugSequence ( target_size=target_size, crop_range=crop_range, allow_hor_flip=True, batch_size=128, subtractMean=0.0, datasrc=datasrc, test=False )

# First go over the data set to get the RGB means
m = 0
i = 0
RGB_mean = np.array ( [ 0., 0., 0. ] )
for X,Y in dataGen:    
    m_batch = X.shape[0]
    RGB_mean_batch = np.mean ( np.mean ( np.mean (X, axis=2), axis=1), axis=0) #mean accross all training example, height, width
    #print (RGB_mean_batch.shape, RGB_mean.shape, RGB_mean_batch, RGB_mean )
    RGB_mean = ( RGB_mean * m + RGB_mean_batch * m_batch ) / (m + m_batch)
    m += m_batch
    i +=1
    print ("iter", i, "/", len(dataGen), "; RGB_mean:", RGB_mean)

print ("Final RGB_mean:", RGB_mean)

# Second pass through data to get covariance R, G, B
m = 0
i = 0
RGB_covar = np.zeros((3,3))
for X,Y in dataGen:    
    m_batch = X.shape[0]
    cntpix_batch = X.shape[0] * X.shape[1] * X.shape[2]
    # X shape is [m,255,255,3]; RGB_mean shape is [,3]. 
    # NumPy ... starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when they are equal
    X_centered = X - RGB_mean
    
    R_centered_batch = np.ravel (X_centered[:,:,:,0])
    G_centered_batch = np.ravel (X_centered[:,:,:,1])
    B_centered_batch = np.ravel (X_centered[:,:,:,2])
    #print ("R,G,B centerd mean: ", np.mean(R_centered_batch), np.mean(G_centered_batch), np.mean(B_centered_batch))

    RG_covar_batch = np.sum ( np.multiply ( R_centered_batch, G_centered_batch ) ) / cntpix_batch
    RB_covar_batch = np.sum ( np.multiply ( R_centered_batch, B_centered_batch ) ) / cntpix_batch
    GB_covar_batch = np.sum ( np.multiply ( G_centered_batch, B_centered_batch ) ) / cntpix_batch
    RR_covar_batch = np.sum ( np.multiply ( R_centered_batch, R_centered_batch ) ) / cntpix_batch
    GG_covar_batch = np.sum ( np.multiply ( G_centered_batch, G_centered_batch ) ) / cntpix_batch
    BB_covar_batch = np.sum ( np.multiply ( B_centered_batch, B_centered_batch ) ) / cntpix_batch
    covar_batch = np.vstack ( ( \
        np.hstack ((RR_covar_batch, RG_covar_batch, RB_covar_batch)), \
        np.hstack ((RG_covar_batch, GG_covar_batch, GB_covar_batch)), \
        np.hstack ((RB_covar_batch, GB_covar_batch, BB_covar_batch)) ) )
    #covar_batch = np.cov ( np.vstack ((R,G,B)) )

    RGB_covar = (RGB_covar * m + covar_batch * m_batch) / (m + m_batch)

    m += m_batch
    i +=1
    print ("iter", i, "/", len(dataGen), "; RGB_covar:\n", RGB_covar)

    #redd = np.ravel(X[:,:,:,0])
    #greenn = np.ravel(X[:,:,:,1])
    #bluee = np.ravel(X[:,:,:,2])
    #print ("np.cov:\n", np.cov ( np.vstack ((redd, greenn, bluee)) ) )

print ("Final RGB_covar:\n", RGB_covar)

# Get eigenvectors and eigenvalues
(eigval, eigvec) = eig (RGB_covar)

print ( "Make sure RGB_covar * eigvec = eigval * eigvec" )
print ( "np.dot (RGB_covar, eigvec): ", np.dot (RGB_covar, eigvec) )
print ( "eigval * eigvec: ", eigval * eigvec )

# Sort eigenvectors by eigenvalue (decreasing)
order_of_eigenvalues = np.flip ( np.argsort (eigval) )
eigvec = eigvec [:, order_of_eigenvalues ]
eigval = eigval [order_of_eigenvalues]

# Save RGB mean, PCA eigenvectors, eigenvalues to a file for future reference when distorting images while training
np.save ( "..\\rgb_mean.npy", RGB_mean)
np.save ( "..\\eigenvectors.npy", eigvec)
np.save ( "..\\eigenvalues.npy", eigval)
# to load:
#   eigvec = np.load("..\\eigenvectors.npy")
#   eigval = np.load("..\\eigenvalues.npy")

