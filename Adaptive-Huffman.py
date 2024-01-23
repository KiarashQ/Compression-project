import logging
import operator
import os
from bitarray import bitarray
from progress.bar import ShadyBar
import collections
import itertools
import math
import numpy as np
from matplotlib import pyplot as plt


NYT = 'NYT'


class Tree:
    def __init__(self, weight, num, data=None):
        self.weight = weight
        self.num = num
        self._left = None
        self._right = None
        self.parent = None
        self.data = data
        self.code = []

    def __repr__(self):
        return "#%d(%d)%s '%s'" % (self.num, self.weight, self.data, self.code)

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left):
        self._left = left
        if self._left:
            self._left.parent = self

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, right):
        self._right = right
        if self._right:
            self._right.parent = self

    def pretty(self, indent_str='  '):
        return ''.join(self.pretty_impl(0, indent_str))

    def pretty_impl(self, level, indent_str):
        if not self._left and not self._right:
            return [indent_str * level, '%s' % self, '\n']
        line = [indent_str * level, '%s' % self, '\n']
        for subtree in (self._left, self._right):
            if isinstance(subtree, Tree):
                line += subtree.pretty_impl(level + 1, indent_str)
        return line

    def search(self, target):
        stack = collections.deque([self])
        while stack:
            current = stack.pop()
            if current.data == target:
                return {'first_appearance': False, 'code': current.code}
            if current.data == NYT:
                nytcode = current.code
            if current.right:
                current.right.code = current.code + [1]
                stack.append(current.right)
            if current.left:
                current.left.code = current.code + [0]
                stack.append(current.left)
        return {'first_appearance': True, 'code': nytcode}


def exchange(node1, node2):
    node1.left, node2.left = node2.left, node1.left
    node1.right, node2.right = node2.right, node1.right
    node1.data, node2.data = node2.data, node1.data


class AdaptiveHuffman:
    def __init__(self, byte_seq, alphabet_range=(0, 255), dpcm=False):
        self.byte_seq = byte_seq
        self.dpcm = dpcm
        self._bits = None
        self._bits_idx = 0
        self._alphabet_first_num = min(alphabet_range)
        alphabet_size = abs(alphabet_range[0] - alphabet_range[1]) + 1
        self.exp = alphabet_size.bit_length() - 1
        self.rem = alphabet_size - 2**self.exp
        self.current_node_num = alphabet_size * 2 - 1
        self.tree = Tree(0, self.current_node_num, data=NYT)
        self.all_nodes = [self.tree]
        self.nyt = self.tree

    def encode(self):
        def encode_fixed_code(dec):
            alphabet_idx = dec - (self._alphabet_first_num - 1)
            if alphabet_idx <= 2 * self.rem:
                fixed_str = '{:0{padding}b}'.format(alphabet_idx - 1, padding=self.exp + 1)
            else:
                fixed_str = '{:0{padding}b}'.format(alphabet_idx - self.rem - 1, padding=self.exp)
            return bin_str2bool_list(fixed_str)

        def bits2bytes(bits):
            return (bits + 7) // 8

        def bin_str2bool_list(binary_string):
            return [c == '1' for c in binary_string]

        progressbar = ShadyBar('encoding', max=len(self.byte_seq), suffix='%(percent).1f%% - %(elapsed_td)ss')

        if self.dpcm:
            self.byte_seq = tuple(encode_dpcm(self.byte_seq))

        logging.getLogger(__name__).info('entropy: %f', entropy(self.byte_seq))

        code = []
        for symbol in self.byte_seq:
            fixed_code = encode_fixed_code(symbol)
            result = self.tree.search(fixed_code)
            if result['first_appearance']:
                code.extend(result['code'])
                code.extend(fixed_code)
            else:
                code.extend(result['code'])
            self.update(fixed_code, result['first_appearance'])
            progressbar.next()

        remaining_bits_length = bits2bytes(len(code) + 3) * 8 - (len(code) + 3)
        code = (bin_str2bool_list('{:03b}'.format(remaining_bits_length)) + code)

        progressbar.finish()
        return bitarray(code)

    def decode(self):
        def read_bits(bit_count):
            progressbar.next(bit_count)
            ret = self._bits[self._bits_idx:self._bits_idx + bit_count]
            self._bits_idx += bit_count
            return ret

        def decode_fixed_code():
            fixed_code = read_bits(self.exp)
            integer = bool_list2int(fixed_code)
            if integer < self.rem:
                fixed_code += read_bits(1)
                integer = bool_list2int(fixed_code)
            else:
                integer += self.rem
            return integer + 1 + (self._alphabet_first_num - 1)

        bits = bitarray()
        bits.frombytes(self.byte_seq)
        self._bits = bits.tolist()
        self._bits_idx = 0

        progressbar = ShadyBar('decoding', max=len(self._bits), suffix='%(percent).1f%% - %(elapsed_td)ss')

        remaining_bits_length = bool_list2int(read_bits(3))
        if remaining_bits_length:
            del self._bits[-remaining_bits_length:]
            progressbar.next(remaining_bits_length)
        self._bits = tuple(self._bits)

        code = []
        while self._bits_idx < len(self._bits):
            current_node = self.tree
            while current_node.left or current_node.right:
                bit = read_bits(1)[0]
                current_node = current_node.right if bit else current_node.left
            if current_node.data == NYT:
                first_appearance = True
                dec = decode_fixed_code()
                code.append(dec)
            else:
                first_appearance = False
                dec = current_node.data
                code.append(current_node.data)
            self.update(dec, first_appearance)

        progressbar.finish()
        return decode_dpcm(code) if self.dpcm else code

    def update(self, data, first_appearance):
        def find_node_data(data):
            for node in self.all_nodes:
                if node.data == data:
                    return node
            raise KeyError(f'Cannot find the target node given {data}.')

        current_node = None
        while True:
            if first_appearance:
                current_node = self.nyt
                self.current_node_num -= 1
                new_external = Tree(1, self.current_node_num, data=data)
                current_node.right = new_external
                self.all_nodes.append(new_external)

                self.current_node_num -= 1
                self.nyt = Tree(0, self.current_node_num, data=NYT)
                current_node.left = self.nyt
                self.all_nodes.append(self.nyt)

                current_node.weight += 1
                current_node.data = None
                self.nyt = current_node.left
            else:
                if not current_node:
                    current_node = find_node_data(data)
                node_max_num_in_block = max(
                    (
                        n for n in self.all_nodes
                        if n.weight == current_node.weight
                    ),
                    key=operator.attrgetter('num')
                )
                if node_max_num_in_block not in (current_node, current_node.parent):
                    exchange(current_node, node_max_num_in_block)
                    current_node = node_max_num_in_block
                current_node.weight += 1
            if not current_node.parent:
                break
            current_node = current_node.parent
            first_appearance = False


def compress(in_filename, out_filename, alphabet_range, dpcm):
    with open(in_filename, 'rb') as in_file:
        logging.getLogger(__name__).info('open file: "%s"', in_filename)
        content = in_file.read()
        logging.getLogger(__name__).info(
            'original size: %d bytes', os.path.getsize(in_file.name)
        )
    ada_huff = AdaptiveHuffman(content, alphabet_range, dpcm)
    code = ada_huff.encode()

    with open(out_filename, 'wb') as out_file:
        logging.getLogger(__name__).info('write file: "%s"', out_filename)
        code.tofile(out_file)
    logging.getLogger(__name__).info(
        'compressed size: %d bytes', os.path.getsize(out_filename)
    )


def extract(in_filename, out_filename, alphabet_range, dpcm):
    with open(in_filename, 'rb') as in_file:
        logging.getLogger(__name__).info('open file: "%s"', in_filename)
        content = in_file.read()
        logging.getLogger(__name__).info(
            'original size: %d bytes', os.path.getsize(in_file.name)
        )
    ada_huff = AdaptiveHuffman(content, alphabet_range, dpcm)
    code = ada_huff.decode()

    with open(out_filename, 'wb') as out_file:
        logging.getLogger(__name__).info('write file: "%s"', out_filename)
        out_file.write(bytes(code))
    logging.getLogger(__name__).info(
        'extract size: %d bytes', os.path.getsize(out_filename)
    )


def show_raw_img(original_filename, extracted_filename, size):
    with open(original_filename, 'rb') as img_file:
        original_img = np.fromfile(img_file, dtype=np.uint8)
    with open(extracted_filename, 'rb') as img_file:
        extracted_img = np.fromfile(img_file, dtype=np.uint8)
    _, axarr = plt.subplots(1, 2)
    axarr[0].set_title('Original')
    axarr[0].imshow(original_img, cmap='gray')
    axarr[1].set_title('After Compression and Extraction')
    axarr[1].imshow(extracted_img, cmap='gray')
    plt.show()


def encode_dpcm(seq):
    return (
        (item - seq[idx - 1]) & 0xff if idx else item
        for idx, item in enumerate(seq)
    )


def decode_dpcm(seq):
    return itertools.accumulate(seq, lambda x, y: (x + y) & 0xff)


def bin_str2bool_list(binary_string):
    return [c == '1' for c in binary_string]


def bool_list2bin_str(boolean_list):
    return ''.join('1' if i else '0' for i in boolean_list)


def bool_list2int(boolean_list):
    return sum(v << i for i, v in enumerate(reversed(boolean_list)))


def entropy(byte_seq):
    counter = collections.Counter(byte_seq)
    ret = 0
    for count in counter.values():
        prob = count / sum(counter.values())
        ret += prob * math.log2(prob)
    return -ret


if __name__ == "__main__":
    # Example usage
    in_filename = "2_img_.raw"
    out_filename = "2_img_compressed_adaptive_huffman1.bin"
    extracted_filename = "2_img_decompressed_adaptive_huffman1.raw"

    alphabet_range = (0, 255)
    dpcm = False

    compress(in_filename, out_filename, alphabet_range, dpcm)
    extract(out_filename, extracted_filename, alphabet_range, dpcm)
