import bitarray
import filecmp
import numpy as np

class Nodes:
    def __init__(self, probability, symbol, left=None, right=None):
        self.probability = probability
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

def CalculateProbability(input_array, steps):
    the_symbols = dict()
    for i in range(0, len(input_array), steps):
        symbol = input_array[i:i+ steps].to01()
        if the_symbols.get(symbol) is None:
            the_symbols[symbol] = 1
        else:
            the_symbols[symbol] += 1
    return the_symbols

the_codes = dict()

def CalculateCodes(node, value=''):
    newValue = value + str(node.code)

    if (node.left):
        CalculateCodes(node.left, newValue)
    if (node.right):
        CalculateCodes(node.right, newValue)

    if (not node.left and not node.right):
        the_codes[node.symbol] = newValue

    return the_codes

def OutputEncoded(input_array, coding, steps):
    encodingOutput = []
    for i in range(0, len(input_array), steps):
        symbol = input_array[i:i + steps].to01()
        #print(coding[symbol], end='')
        encodingOutput.append(coding[symbol])

    the_string = ''.join([str(item) for item in encodingOutput])
    return the_string


def HuffmanEncoding(input_array, steps):
    symbolWithProbs = CalculateProbability(input_array, steps)
    the_symbols = symbolWithProbs.keys()
    the_probabilities = symbolWithProbs.values()


    the_nodes = []

    for symbol in the_symbols:
        the_nodes.append(Nodes(symbolWithProbs.get(symbol), symbol))

    while len(the_nodes) > 1:
        the_nodes = sorted(the_nodes, key=lambda x: x.probability)
        right = the_nodes[0]
        left = the_nodes[1]
        left.code = 0
        right.code = 1

        newNode = Nodes(left.probability + right.probability, left.symbol + right.symbol, left, right)

        the_nodes.remove(left)
        the_nodes.remove(right)
        the_nodes.append(newNode)

    huffmanEncoding = CalculateCodes(the_nodes[0])
    encoded_output = OutputEncoded(input_array, huffmanEncoding, steps)
    return encoded_output, the_nodes[0]


def HuffmanDecoding(encoded_image, huffmanTree):
    treeHead = huffmanTree
    decodedOutput = []
    for x in encoded_image:
        if x == '1':
            huffmanTree = huffmanTree.right
        elif x == '0':
            huffmanTree = huffmanTree.left

        if huffmanTree.left is None and huffmanTree.right is None:
            decodedOutput.append(huffmanTree.symbol)
            huffmanTree = treeHead

    string = ''.join([str(item) for item in decodedOutput])
    return string

def uint8_to_bit_string(uint8_array):
    return ''.join(format(byte, '08b') for byte in uint8_array)

def bit_string_to_uint8(bit_string):
    return np.array([int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)], dtype='uint8')


def encoder(array):
    steps = 8
    input_bit_string = uint8_to_bit_string(array)

    image_samples = bitarray.bitarray(input_bit_string)
    encoded_output, tree = HuffmanEncoding(image_samples, steps)

    encoded_bytes = bitarray.bitarray(encoded_output).tobytes()
    return encoded_bytes, tree

def decoder(compressed_data, tree):
    bit_array = bitarray.bitarray()
    bit_array.frombytes(compressed_data)
    decoded_output = HuffmanDecoding(bit_array.to01(), tree)
    decoded_bytes = bitarray.bitarray(decoded_output).tobytes()
    return decoded_bytes



def main():
    steps = 8
    input_image_path = "2_img_.raw"
    compressed_path = "2_img_compressed_huffman1.bin"
    decompressed_path = "2_img_decompressed_huffman1.raw"

    # Read Image as uint8
    image_samples = np.fromfile(input_image_path, dtype='uint8')

    input_bit_string = uint8_to_bit_string(image_samples)
    image_samples = bitarray.bitarray(input_bit_string)

    encoded_output, the_tree = HuffmanEncoding(image_samples, steps)
    encoded_bytes = bitarray.bitarray(encoded_output).tobytes()
    with open(compressed_path, 'wb') as compressed_file:
        compressed_file.write(encoded_bytes)

    # ------------------------------------------------------

    # Read the compressed file
    compressed_file = open(compressed_path, 'rb')
    compressed_data = bitarray.bitarray()
    compressed_data.fromfile(compressed_file)
    compressed_file.close()

    # Decompression
    decoded_output = HuffmanDecoding(compressed_data.to01(), the_tree)
    decoded_bytes = bitarray.bitarray(decoded_output).tobytes()
    with open(decompressed_path, 'wb') as decompressed_file:
        decompressed_file.write(decoded_bytes)

    print(f"Files are identical? {filecmp.cmp(input_image_path, decompressed_path)}")


if __name__ == '__main__':
    main()
