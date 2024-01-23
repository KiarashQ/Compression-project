import numpy as np

def rle_encode(data):
    encoding = []
    prev_char = data[0]
    count = 1

    for char in data[1:]:
        if char == prev_char and count < 255:
            count += 1
        else:
            encoding.append((prev_char, count))
            prev_char = char
            count = 1
    encoding.append((prev_char, count))
    return encoding


def rle_decode(data):
    ''' Decode RLE '''
    decoded_data = []
    for char, count in data:
        decoded_data.extend([char] * count)
    return decoded_data

def read_binary_file(file_path):
    ''' Read binary file '''
    with open(file_path, "rb") as file:
        return np.fromfile(file, dtype=np.uint8)

def write_binary_file(data, file_path):
    ''' Write binary file '''
    with open(file_path, "wb") as file:
        file.write(bytearray(data))
def write_encoded_data(encoding, file_path):
    with open(file_path, "wb") as file:
        for char, count in encoding:
            file.write(bytearray([char]))
            file.write(count.to_bytes(2, 'little'))  # Using 2 bytes for count

def read_encoded_data(file_path):
    with open(file_path, "rb") as file:
        data = file.read()
        return [(data[i], int.from_bytes(data[i+1:i+3], 'little')) for i in range(0, len(data), 3)]



# Example usage
source_file = "2-img_pred_quant128.bin"
encoded_file = "encoded_the_128_with_rle.bin"
decoded_file = "decoded_the_128_with_rle.bin"

# Encode
original_data = read_binary_file(source_file)
encoded_data = rle_encode(original_data)
write_encoded_data(encoded_data, encoded_file)

# Decode
encoded_data = read_encoded_data(encoded_file)
decoded_data = rle_decode(encoded_data)
write_binary_file(decoded_data, decoded_file)

