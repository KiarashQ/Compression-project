import imageio.v2 as imageio
import numpy as np
import os

def png_to_raw_bsq(input_png_path, output_raw_path):
    # Read the image using imageio v2
    img = imageio.imread(input_png_path)

    # Ensure the image is in the correct format (RGB or RGBA)
    if img.ndim == 2:  # Grayscale image
        img = np.stack((img, img, img), axis=-1)  # Convert to RGB
    elif img.shape[2] == 4:  # RGBA image
        img = img[:, :, :3]  # Drop alpha channel

    # Convert image to BSQ format
    bsq_img = img.transpose(2, 0, 1)  # Transpose to (bands, height, width)

    # Write to a raw file
    with open(output_raw_path, 'wb') as file:
        file.write(bsq_img.tobytes())

# Example usage
sourcedir = "50 PNG"
outputdir = "Converted_RAW"

# Create output directory if it doesn't exist
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

dl = [f for f in os.listdir(sourcedir) if f.endswith(".png")]

for f in dl:
    filename_without_ext = os.path.splitext(f)[0]
    input_png_path = os.path.join(sourcedir, f)
    output_raw_path = os.path.join(outputdir, filename_without_ext + ".raw")
    png_to_raw_bsq(input_png_path, output_raw_path)
#input_png_path = '4.png'  # Replace with your PNG file path
#output_raw_path = '4.raw'  # Replace with your desired output file path

