import decimal
from decimal import Decimal
from collections import Counter
import filecmp

decimal.getcontext().prec = 10


class ArithmeticEncoding:
    def __init__(self, frequency_table):
        self.probability_table = self.get_probability_table(frequency_table)
        self.stage_probs = self.calculate_stage_probs()

    def get_probability_table(self, frequency_table):
        total_frequency = sum(frequency_table.values())
        return {key: Decimal(value) / total_frequency for key, value in frequency_table.items()}

    def calculate_stage_probs(self):
        stage_probs = {}
        stage_min = Decimal(0.0)
        for symbol, prob in self.probability_table.items():
            stage_max = stage_min + prob
            stage_probs[symbol] = (stage_min, stage_max)
            stage_min = stage_max
        return stage_probs

    def encode(self, message):
        stage_min, stage_max = Decimal(0.0), Decimal(1.0)
        for symbol in message:
            symbol_min, symbol_max = self.stage_probs[symbol]
            stage_min, stage_max = stage_min + symbol_min * (stage_max - stage_min), stage_min + symbol_max * (stage_max - stage_min)
        return (stage_min + stage_max) / 2

    def decode(self, encoded_value, message_length):
        decoded_message = []
        stage_min, stage_max = Decimal(0.0), Decimal(1.0)
        for _ in range(message_length):
            for symbol, (symbol_min, symbol_max) in self.stage_probs.items():
                if stage_min + symbol_min * (stage_max - stage_min) <= encoded_value < stage_min + symbol_max * (stage_max - stage_min):
                    decoded_message.append(symbol)
                    stage_min, stage_max = stage_min + symbol_min * (stage_max - stage_min), stage_min + symbol_max * (stage_max - stage_min)
                    break
        return decoded_message


def read_raw_image(file_path, chunk_size):
    with open(file_path, 'rb') as file:
        data = file.read()
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def write_to_file(data, file_path):
    with open(file_path, 'wb') as file:
        file.write(data)


def read_binary_file(file_path):
    """Reads binary data from a file."""
    with open(file_path, 'rb') as file:
        return file.read()


def decimal_to_bytes(decimal_value):
    # Convert Decimal to a string, then encode the string to bytes
    decimal_str = str(decimal_value)
    return decimal_str.encode('utf-8')


def bytes_to_decimal(byte_data):
    # Decode the bytes back to a string, then convert to Decimal
    decimal_str = byte_data.decode('utf-8')
    return Decimal(decimal_str)


def test_encoding_decoding():
    test_message = ['A', 'B', 'A', 'C']
    test_frequency_table = Counter(test_message)

    # Initialize encoder with test frequency table
    encoder = ArithmeticEncoding(test_frequency_table)

    # Encode the test message
    encoded_value = encoder.encode(test_message)

    # Decode the encoded message
    decoded_message = encoder.decode(encoded_value, len(test_message))

    # Check if decoded message matches original
    assert test_message == decoded_message, "Decoded message does not match original"
    print("Test Passed: Decoded message matches original")


def test_edge_cases():
    test_message = ['A', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']  # 'A' and 'B' are low probability
    test_frequency_table = Counter(test_message)

    # Initialize encoder with test frequency table
    encoder = ArithmeticEncoding(test_frequency_table)

    # Encode the test message
    encoded_value = encoder.encode(test_message)

    # Decode the encoded message
    decoded_message = encoder.decode(encoded_value, len(test_message))

    # Check if decoded message matches original
    assert test_message == decoded_message, "Decoded message does not match original in edge case test"
    print("Edge Case Test Passed: Decoded message matches original")

test_edge_cases()

test_encoding_decoding()

def main():
    input_image_path = "57_img_.raw"
    compressed_path = "57_img_compressed.bin"
    decompressed_path = "57_img_decompressed.raw"
    chunk_size = 1  # Adjust based on your image's color depth

    input_chunks = read_raw_image(input_image_path, chunk_size)
    frequency_table = Counter(input_chunks)

    arithmetic_encoder = ArithmeticEncoding(frequency_table)
    encoded_value = arithmetic_encoder.encode(input_chunks)

    compressed_data = decimal_to_bytes(encoded_value)
    write_to_file(compressed_data, compressed_path)

    compressed_data_bytes = read_binary_file(compressed_path)
    compressed_data_decimal = bytes_to_decimal(compressed_data_bytes)

    decoded_chunks = arithmetic_encoder.decode(compressed_data_decimal, len(input_chunks))
    decompressed_data = b''.join(decoded_chunks)
    write_to_file(decompressed_data, decompressed_path)


    print(f"Files are identical? {filecmp.cmp(input_image_path, decompressed_path)}")


if __name__ == "__main__":
    main()
