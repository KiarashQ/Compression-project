import numpy as np

def msecalc(array1,array2):
    """Calculates Mean Square Error between two arrays of bytes"""
    return np.mean((array1 - array2)**2)      # Mean Squared Error Formmula

def PSNRcalc(array1, array2):
    """Calculates Peak Signal to Noise Ratio"""
    mse = msecalc(array1,array2)
    max_value = 255             # Assuming 8-bit             
    return 20*np.log10(max_value / np.sqrt(mse))  


if __name__ == "__main__":

    file1 = "2_img_.raw"          # Original Image
    file2 = "2-img_pred_quant_decompressed128.raw"
    raw_bytes1 = np.fromfile(file1,dtype='uint8')
    raw_bytes2 = np.fromfile(file2,dtype='uint8')

    print(f"Mean Square Error (MSE) : {msecalc(raw_bytes1,raw_bytes2):.4f}")
    print(f"Peak S-N    Ratio (PSNR): { PSNRcalc(raw_bytes1,raw_bytes2):.4f} dB")

