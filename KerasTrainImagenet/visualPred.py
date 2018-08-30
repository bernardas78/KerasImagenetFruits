from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

from DataGen import DataGen_v1_150x150_1frame as dg_v1 

# Run:
#   visualCache = vp.visualInit() #to initialize plot
#   vp.visualShow(visualCache)

def visualInit():
    # Visualizes pictures randomly; waits for key-pres between visualizations
    #
    model = load_model ("C:\\labs\models\\model_v22.h5")

    dataGen = dg_v1.prepDataGen()

    # New plot
    rows=8
    columns=4
    fig = plt.figure ( figsize = (columns*2, rows ) )
    #fig.patch.set_visible(False)
    ims=[]
    ims_data={}

    #Remove margins
    plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)

    #Plot in interactive mode - does not block command line
    plt.ion()

    for imgind in range(rows*columns):
        subplot = fig.add_subplot(rows, columns*2, imgind*2+1)
        #No labels and markings on axis
        _ = subplot.set_xticklabels([])
        _ = subplot.set_yticklabels([])
        _ = subplot.set_xticks([])
        _ = subplot.set_yticks([])
        #add image to array (iteratively will change data of this for speed rather than replacing image)
        im = subplot.imshow(np.random.rand(150,150,3) ) #, cmap='gray')
        ims = np.append(ims, im)

        subplot_lbl = fig.add_subplot(rows, columns*2, imgind*2+2)
        #No labels and markings on axis
        _ = subplot_lbl.set_xticklabels([])
        _ = subplot_lbl.set_yticklabels([])
        _ = subplot_lbl.set_xticks([])
        _ = subplot_lbl.set_yticks([])

        # Predicted labels and percentages
        #pred1 = subplot_lbl.text(x=0.5,y=0.8,s="pred1", ha="center", va="center", fontsize=8, color="red")
        #ims_data[ "pred1." + str(imgind) ] = pred1
        values = np.random.rand(5)
        values_lbl = [("%.2f" % value) for value in values]
        pred_bars = subplot_lbl.barh ( y = np.arange(5), width = values) #, tick_label = values_lbl )
        for bar_ind in np.arange(len(pred_bars)):
            bar = pred_bars[bar_ind]
            width = bar.get_width()
            xloc = 0.98 * width
            #clr = 'red'
            #align = 
            yloc = bar.get_y() + bar.get_height()/2.0
            #print("xloc, yloc, values_lbl[bar_ind]",xloc, yloc, values_lbl[bar_ind])
            label = subplot_lbl.text(xloc, yloc, values_lbl[bar_ind], horizontalalignment='right',
                                     verticalalignment='center', color='black', fontsize=6,clip_on=True)

        # Correct label
        #sw = subplot_lbl.get_width()
        #sh = subplot_lbl.get_height()
        #print ("sw, sh:",sw, sh)
        lbl = subplot_lbl.text(x=0.2,y=0,s="actual", ha="left", va="center", fontsize=8)
        ims_data[ "lbl." + str(imgind) ] = lbl
        ims_data[ "lbl_sub." + str(imgind) ] = subplot_lbl

    plt.show()

    cache = {}
    cache ["ims"] = ims
    cache ["ims_data"] = ims_data
    cache ["fig"] = fig
    cache ["rows"] = rows
    cache ["columns"] = columns
    cache ["dataGen"] = dataGen
    cache ["model"] = model

    return cache

    
def visualShow(cache):    

    ims = cache ["ims"]
    ims_data = cache ["ims_data"]
    fig = cache ["fig"]
    rows = cache ["rows"]
    columns = cache ["columns"]
    dataGen = cache ["dataGen"]
    model = cache ["model"]

    class_names = np.array(list(dataGen.class_indices.keys()))

    for X, y in dataGen:

        if model is not None:
            # Get predictions
            yhat = model.predict(X)
            
            # Get order of predicted classes for all training examples (highest-prob classes - first)
            #  shape: [ m, c ]
            yhat_order = np.flip ( np.argsort (yhat, axis=1), axis=1 )
            y_classes = np.argmax( y, axis=1 )

        
        for i in range(rows*columns):
            # Show image on the left
            im = ims [ i ]
            im.set_data( X[i] )

            #Show actual label and predictions on the right
            subplot_lbl = ims_data [ "lbl_sub."+str(i) ]
            subplot_lbl.cla ()
            subplot_lbl.autoscale(False)
            _ = subplot_lbl.set_xticklabels([])
            _ = subplot_lbl.set_yticklabels([])
            _ = subplot_lbl.set_xticks([])
            _ = subplot_lbl.set_yticks([])

            top5_classes = yhat_order [ i ,0:5 ] #reshape to remove last dimension
            top5_yhat_values = yhat [ i, top5_classes ]
            top5_class_lbls = class_names [ top5_classes ]
            top5_yhat_text = [ (top5_class_lbls [indx] + ": %.2f" % top5_yhat_values [indx] ) for indx in np.arange(5)]
            top5_istrue = np.equal( top5_classes, y_classes[i], dtype=int ).astype(int)
            top5_colors = np.array(["orange","green"]) [ top5_istrue ]

            pred_bars = subplot_lbl.barh ( y = (5-np.arange(5))/10.+.4, width = top5_yhat_values, height=0.1, color=top5_colors  )

            # Get subplot width: it will be it's first element's width
            subplot_width = width = pred_bars[0].get_width()

            for bar_ind in np.arange(len(pred_bars)):
                bar = pred_bars[bar_ind]
                width = bar.get_width()
                align="left"
                xloc = 0.02
                yloc = bar.get_y() + bar.get_height()/2.0
                _ = subplot_lbl.text(xloc, yloc, top5_yhat_text[bar_ind], horizontalalignment=align,
                                     verticalalignment='center', color='black', fontsize=8,clip_on=True)

            #Add actual label (in case it does not show in top5 list)
            lbl = subplot_lbl.text(x=0.5,y=0.,s=class_names[y_classes[i]], ha="center", va="bottom", fontsize=10)

        
        fig.canvas.flush_events()
        plt.savefig("C:\\labs\\KerasImagenetFruits\\Visuals\\top5_"+str(dataGen.batch_index)+".jpg")
        break

def visualClose():
    plt.close()