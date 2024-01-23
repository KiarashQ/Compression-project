# Prediction and quantization
import numpy as np

source_file = "Converted_RAW/" + "2_img_.raw" 

output_file = 'Quantized/' + '2_img_q_p.raw' 

def prediction(data):               # Predict the previous Value
    predicted_value = 0             # First value to predict will be zero
    delta = []
    for I in data:    
        image_sample = int(I)       # Since Raw is in uint8, we need to change to int in order to represent negative errors       
        delta.append(image_sample - predicted_value)   # Delta = I - P    
        predicted_value = I                            # Store the last image sample 
    return delta

def inverse_prediction(data):       # Prediction is based on the last value of the reconstructed image.
    predicted_value = 0
    reconstructed_image = []
    for delta in data:
        reconstructed_sample = delta + predicted_value
        reconstructed_image.append(reconstructed_sample)
        predicted_value = reconstructed_sample  
    return reconstructed_image

def quantization(data,qstep):
    data = np.array(data)               # If inuput a list, convert to array.
    return data//qstep

def dequantization_mp(data,qstep):      # Mid Point Reconstruction 
    data = np.array(data)   
    return data*qstep + qstep//2

def entropy_calculation(image_data):
    _, counts = np.unique(image_data, return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def print_proof(data): 
    quantization_indexes = quantization(data,3)   # Quantize
    prediction_res = prediction(quantization_indexes)      # Predict
    inversed_prediction = inverse_prediction(prediction_res)         # Un-Predict
    reconstructed_image = dequantization_mp(inversed_prediction,3)   # De-quantize
    print("---QUANTIZATION + PREDICTION Process - Encoder Side:")
    print("\nI:             ",data) 
    print("\nQ(I):        ",quantization_indexes)
    print("Pred residuals:",prediction_res)
    print(" \n- TRANSMISSION PHASE - Decoder Side\n ")
    print("Inv Prediction:",inversed_prediction)
    print("\nI'              ",reconstructed_image)
    print("\nError= I-I':    ",data-reconstructed_image)

def quant_and_pred(image_samples,qstep):
    quantization_indexes = quantization(image_samples,qstep)   # Quantize
    return np.array(prediction(quantization_indexes) )         # Predict

def pred_quant_reconstruct(prediction_res,qstep):
    inversed_prediction = inverse_prediction(prediction_res)
    return np.array(dequantization_mp(inversed_prediction,qstep),dtype='uint8')
    

# if __name__ == "__main__":
    # image_samples = np.fromfile(source_file,dtype='uint8') # Read Image
    # quantization_and_prediction(image_samples)       # Quantize, Predict, and reconstruct after.
    
    