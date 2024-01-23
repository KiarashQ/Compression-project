import numpy as np
import filecmp

def calculate_entropy(image_array):
    _, counts = np.unique(image_array, return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def main():
    source_file = '2-img_pred_quant128.bin'
    source_file1 = "quantized by me.raw"
    source_file2 = '2-img_pred_quant_decompressed128.raw'

    original_image = np.fromfile(source_file,dtype='uint8') # Read Image
    quantized_image = np.fromfile(source_file1,dtype='uint8') # Read Image
    predicted_quantized_image = np.fromfile(source_file2,dtype='uint8') # Read Image

    # Calculate and print entropies
    original_entropy = calculate_entropy(original_image)
    quantized_entropy = calculate_entropy(quantized_image)
    predicted_quantized_entropy = calculate_entropy(predicted_quantized_image)
    print(f"Original Entropy: {original_entropy:.2f}")
    print(f"Quantized Entropy: {quantized_entropy:.2f}")
    print(f"Predicted and Quantized Entropy: {predicted_quantized_entropy:.2f}")

    print(f"Files are identical? {filecmp.cmp(source_file, source_file2)}")


if __name__ == '__main__':
    main()