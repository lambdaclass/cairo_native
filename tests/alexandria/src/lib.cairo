mod alexandria {
    // Alexandria Math

    fn fib() -> felt252 {
        alexandria_math::fibonacci::fib(16, 10, 1)
    }

    fn karatsuba() -> u128 {
        alexandria_math::karatsuba::multiply(3754192357923759273591, 18492875)
    }

    fn armstrong_number() -> bool {
        alexandria_math::armstrong_number::is_armstrong_number(472587892687682)
    }

    fn aliquot_sum() -> u128 {
        alexandria_math::aliquot_sum::aliquot_sum(67472587892687682)
    }

    fn collatz_sequence() -> Array<u128> {
        alexandria_math::collatz_sequence::sequence(4332490568290368)
    }

    fn extended_euclidean_algorithm() -> (u128, u128, u128) {
        alexandria_math::extended_euclidean_algorithm::extended_euclidean_algorithm(
            384292543952858, 158915958590
        )
    }

    // Alexandria Data Structures

    use alexandria_data_structures::vec::{Felt252Vec, VecTrait};
    fn vec() -> (felt252, felt252, felt252) {
        let mut vec = VecTrait::<Felt252Vec, felt252>::new();
        vec.push(12);
        vec.push(99);
        vec.set(1, 67);
        (vec.at(0), vec.at(1), vec.at(2))
    }

    use alexandria_data_structures::stack::{Felt252Stack, StackTrait};
    fn stack() -> (Option<felt252>, Option<felt252>, Option<felt252>, bool) {
        let mut stack = StackTrait::<Felt252Stack, felt252>::new();
        stack.push(1);
        stack.push(2);
        stack.push(17);
        let top = stack.peek();
        stack.pop();
        (top, stack.pop(), stack.pop(), stack.is_empty())
    }

    use alexandria_data_structures::queue::{Queue, QueueTrait};
    fn queue() -> (Option<felt252>, Option<felt252>, Option<felt252>, bool) {
        let mut queue = QueueTrait::<felt252>::new();
        queue.enqueue(3);
        queue.enqueue(31);
        queue.enqueue(13);
        (queue.dequeue(), queue.dequeue(), queue.dequeue(), queue.is_empty())
    }

    use alexandria_data_structures::bit_array::{BitArray, BitArrayTrait};
    fn bit_array() -> Option<felt252> {
        let mut bit_array: BitArray = Default::default();
        bit_array.write_word_be(340282366920938463463374607431768211455, 128);
        bit_array.pop_front();
        bit_array.append_bit(true);
        bit_array.append_bit(false);
        bit_array.append_bit(true);
        bit_array.read_word_le(bit_array.len())
    }

    // Alexandria Encoding
    use alexandria_encoding::base64::{Encoder, Decoder, Base64Encoder, Base64Decoder};
    use core::array::ArrayTrait;
    fn base64_encode() -> (Array<u8>, Array<u8>) {
        let mut input = ArrayTrait::<u8>::new();
        input.append('C');
        input.append('a');
        input.append('i');
        input.append('r');
        input.append('o');
        let encoded = Base64Encoder::encode(input);
        let decoded = Base64Decoder::decode(encoded.clone());
        (encoded, decoded)
    }
// Compiling the following functions generates invalid MLIR, please uncomment once the bug is fixed

// use alexandria_encoding::reversible::{U16ReversibleBits, U32ReversibleBits, U64ReversibleBits, U128ReversibleBits, U256ReversibleBits};
// fn reverse_bits() -> (u16, u32, u64, u128, u256) {
//     (
//         U16ReversibleBits::reverse_bits(@333),
//         U32ReversibleBits::reverse_bits(@3333333),
//         U64ReversibleBits::reverse_bits(@3333333333333),
//         U128ReversibleBits::reverse_bits(@333333333333333333333),
//         U256ReversibleBits::reverse_bits(@33333333333333333333333333),
//     )
// }

// use alexandria_encoding::reversible::{U16Reversible, U32Reversible, U64Reversible, U128Reversible, U256Reversible};
// fn reverse_bytes() -> (u16, u32, u64, u128, u256) {
//     (
//         U16Reversible::reverse_bytes(@333),
//         U32Reversible::reverse_bytes(@3333333),
//         U64Reversible::reverse_bytes(@3333333333333),
//         U128Reversible::reverse_bytes(@333333333333333333333),
//         U256Reversible::reverse_bytes(@33333333333333333333333333),
//     )
// }
}
