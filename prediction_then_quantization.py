# Prediction and quantization
import numpy as np

source_file = "2_img_.raw"
output_file1 = '2-img_p_q.bin'
output_file = '2_img_p_q.raw'


def quantization(data,qstep):
    data = np.array(data)               # If inuput a list, convert to array.
    return data//qstep

def dequantization_mp(data,qstep):      # Mid Point Reconstruction 
    data = np.array(data)   
    return data*qstep + qstep//2


def prediction_and_quantization(image_data,qstep):               # Predict the previous Value
    predicted_value = 0             # First value to predict will be zero
    delta = []
    quantized_delta_ls = []
    for I in image_data:    
        image_sample = int(I)       # Since Raw is in uint8, we need to change to int in order to represent negative errors       
        delta = image_sample - predicted_value
        quantized_delta = quantization(delta,qstep)
        quantized_delta_ls.append(quantized_delta)
        reconstructed_delta = dequantization_mp(quantized_delta,qstep)
        reconstructed_sample = reconstructed_delta + predicted_value
        predicted_value = reconstructed_sample                           # Store the last image sample 
    return quantized_delta_ls

def inverse_quantization_and_prediction(received_delta,qstep):       # Prediction is based on the last value of the reconstructed image.
    predicted_value = 0
    reconstructed_image = []
    for delta in received_delta:
        reconstructed_delta = dequantization_mp(delta,qstep)
        reconstructed_sample = predicted_value + reconstructed_delta
        reconstructed_image.append(reconstructed_sample)
        predicted_value = reconstructed_sample  
    return reconstructed_image

def entropy_calculation(image_data):
    _, counts = np.unique(image_data, return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


if __name__ == "__main__":
    qstep = 128
    image_samples = np.fromfile(source_file,dtype='uint8') # Read Image
    quantized_delta = prediction_and_quantization(image_samples,qstep)

    transferable_image = np.array(quantized_delta, dtype='uint8')
    transferable_image.tofile(output_file1)

    reconstructed_image = np.array(inverse_quantization_and_prediction(quantized_delta,qstep),dtype='uint8')
    reconstructed_image.tofile(output_file)
    print("Original Entropy:", entropy_calculation(image_samples))
    print("Pred + Q Entropy:", entropy_calculation(quantized_delta))