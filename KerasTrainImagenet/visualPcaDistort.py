# Perform visual image distortion and save results:
#   Each Principal Component is added proportional to it's eigenvalue (significance) in the new coordinate system
#   RGB += [PC1; PC2; PC3] * [eigenval1 * alpha1; eigenval2 * alpha2; eigenval3 * alpha3].
#       where [alpha1; alpha2; alpha3] values from array [-1.0, -.3, -.1, -.03, -.01, .01, .03, .1, .3, 1.0]

# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("visualPcaDistort.py").read())

import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

# Load eigenvectors and -values (saved in ..\\KerasImagenetFruits\\PCA.py")
eigvec = np.load("..\\eigenvectors.npy")
eigval = np.load("..\\eigenvalues.npy")

img_paths = [ \
    "n01531178\\n01531178_1980.JPEG",\
    "n01494475\\n01494475_103.JPEG",\
    "n01496331\\n01496331_318.JPEG",\
    "n01514668\\n01514668_19.JPEG",\
    #"n01518878\n01518878_103.JPEG",\
    #"n01530575\n01530575_5.JPEG",\
    "n01530575\\n01530575_114.JPEG",\
    "n01531178\\n01531178_431.JPEG",\
    "n01532829\\n01532829_145.JPEG",\
    "n01537544\\n01537544_108.JPEG"]
list_alpha_values = [-1.0, -.3, -.1, -.03, -.01, .01, .03, .1, .3, 1.0]

for img_ind in range(len(img_paths)):
    im = np.asarray ( Image.open( "C:\\ILSVRC14\\ILSVRC2012_img_train_unp_50\\" + img_paths[img_ind] ) ) / 255.
    #im.save( )
    #scipy.misc.imsave( "C:\\Users\\bciap\\Desktop\\PCA\\img" + str(img_ind) + "_original.jpeg", im )

    fig = plt.figure( figsize = (\
        # 1 column per alpha value + 1 column for header: PC index \
        len (list_alpha_values) + 1,\
        # 3x eigenvectors to distort against + 1 line for header: alpha values
        4 ) )                               
    plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)
    
    subplot = fig.add_subplot ( \
        4, \
        len (list_alpha_values) + 1,\
        1)                                  # Top/left corner: original image
    _ = subplot.set_xticklabels([])
    _ = subplot.set_yticklabels([])
    _ = subplot.set_xticks([])
    _ = subplot.set_yticks([])
    subplot.imshow( im)

    # In left column: show eigenvalues of Principal components
    for pc_index in range(3):
        subplot = fig.add_subplot ( \
            4, \
            len (list_alpha_values) + 1,\
            (pc_index+1) * (len(list_alpha_values)+1) + 1)
        _ = subplot.set_xticklabels([])
        _ = subplot.set_yticklabels([])
        _ = subplot.set_xticks([])
        _ = subplot.set_yticks([])
        subplot.text(0.5, 0.5, "{0:.4f}".format(eigval[pc_index]), horizontalalignment='center', verticalalignment='center', \
            color='black', fontsize=12,clip_on=True)

    # Formula for image distortion: += eigvec * eigval * random_alpha
    for alpha_ind in range(len(list_alpha_values)):
        alpha_val = list_alpha_values[alpha_ind]
        
        # In top row - show alpha values
        subplot = fig.add_subplot ( \
            4, \
            len (list_alpha_values) + 1,\
            alpha_ind + 2)
        _ = subplot.set_xticklabels([])
        _ = subplot.set_yticklabels([])
        _ = subplot.set_xticks([])
        _ = subplot.set_yticks([])
        subplot.text(0.5, 0.5, alpha_val, horizontalalignment='center', verticalalignment='center', \
            color='red' if alpha_val<0 else 'green', fontsize=12,clip_on=True)

        # For each principal component - distort separately
        for pc_index in range (3):
            alpha_arr = np.zeros((3))
            alpha_arr [ pc_index ] = alpha_val

            # += [PC1; PC2; PC3] * [eigenval1 * alpha1; eigenval2 * alpha2; eigenval3 * alpha3].
            summand = np.ravel ( np.dot (eigvec, np.vstack ( np.multiply ( alpha_arr, eigval) ) ) )

            tempim = im+summand*255.

            # Distored image in a subplot
            subplot = fig.add_subplot ( \
                4, \
                len (list_alpha_values) + 1,\
                (pc_index+1) * (len(list_alpha_values)+1) + alpha_ind + 2)
            _ = subplot.set_xticklabels([])
            _ = subplot.set_yticklabels([])
            _ = subplot.set_xticks([])
            _ = subplot.set_yticks([])
            subplot.imshow( tempim)

    plt.savefig ("..\\Visuals\\PcaDistort\\" +str(img_ind)+".jpeg")
