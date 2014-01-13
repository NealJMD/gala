import numpy as np
import sys

def add_derivative_cues(predictions):
    new_preds = []
    predictions = predictions.transpose((2,3,0,1))
    sys.stderr.write(str(predictions.shape) + "\n")
    num_images = len(predictions[:])
    for i in range(0,num_images):
        image = predictions[i,:,:,:]
        num_channels = len(image[:])
        image_list = image.tolist()
        for j in range(0,num_channels):
            bmc = image[j]
            gradients = np.gradient(bmc)
            ygrad = gradients[0]
            xgrad = gradients[1]

            gradient_mags = np.sqrt(ygrad**2 + xgrad**2)
            image_list.append(gradient_mags.tolist())

        new_preds.append(image_list)

    np_new_preds = np.array(new_preds)
    sys.stderr.write(str(np_new_preds.shape) + "\n")
    np_new_preds = np_new_preds.transpose((2,3,0,1))
    sys.stderr.write(str(np_new_preds.shape) + "\n")
    new_chan = np_new_preds[:,:,:,1]
    sys.stderr.write("sum of new channel = " + str(new_chan.sum()) + "\n")
    sys.stderr.write("maxval = " + str(np.amax(new_chan)) + ", minval = " + str(np.amin(new_chan)) + "\n")

    return np_new_preds
