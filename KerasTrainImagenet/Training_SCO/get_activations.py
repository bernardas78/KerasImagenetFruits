from keras.backend import function

def get_layer_activations(model, X, layer):
    #print("Getting activations of X shaped ", X.shape)

    # Initialize a return value: list of activations
    #layer = model.layers[layer_id]

    func_activation = function([model.input], [layer.output])

    output_activation = func_activation([X])[0]

    return output_activation
