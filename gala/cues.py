import numpy as np


def add_derivative_cues(predictions):
    new_preds = []
    predictions = predictions.transpose((0,3,1,2))
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
    np_new_preds = np_new_preds.transpose((0,2,3,1))
    return np_new_preds