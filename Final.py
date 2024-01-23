import os
import numpy as np
from quantization_and_prediction import quant_and_pred, pred_quant_reconstruct, entropy_calculation, quantization, \
    dequantization_mp
from Huff2 import encoder, decoder
import bitarray





source_dir = "Converted_RAW/"                   # Directory where raw data are
output_dir = "compressed/"                  # Directory for compressed data
r_output_dir = "reconstructed/"             # Directory for reconstructed images.
source_files = os.listdir(source_dir)    # Names of the .raw input files.

qstep = 10

def pipeline():
    for file in source_files:

        image_samples = np.fromfile(source_dir+file,dtype='uint8')      # Read raw Data
        # Quantization AND prediction
        delta = quant_and_pred(image_samples, qstep)  # Get prediction delta of quantization indexes

        # Next lines are for quantization ONLY
        #delta = quantization(image_samples, qstep)
        transferable_image = np.array(delta, dtype='uint8')
        inputt, tree = encoder(transferable_image)
        encoded_output = decoder(inputt, tree)
        encoded_array = np.frombuffer(encoded_output, dtype='uint8')

        reconstruction = pred_quant_reconstruct(encoded_array, qstep)  # Reconstruct data after EC^-1
        #reconstruction = dequantization_mp(encoded_array, qstep)

        with open(output_dir + file + '_compressed', 'wb') as new_file:
            new_file.write(inputt)
        with open(r_output_dir + file + '_reconstructed', 'wb') as new_file:
            reconstruction.tofile(new_file)



if __name__ == '__main__':
    pipeline()