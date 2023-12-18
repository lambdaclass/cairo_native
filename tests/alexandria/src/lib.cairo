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
        alexandria_math::extended_euclidean_algorithm::extended_euclidean_algorithm(384292543952858, 158915958590)
    }

    // Alexandria Data Structures

    use alexandria_data_structures::vec::{Felt252Vec, VecTrait};
    fn vec() -> (felt252, felt252, felt252){
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
    fn queue() -> (Option<Box<@felt252>>, Option<felt252>, Option<felt252>, bool){
        let mut queue = QueueTrait::<felt252>::new();
        queue.enqueue(3);
        queue.enqueue(31);
        queue.enqueue(13);
        let front = queue.peek_front();
        queue.dequeue();
        (front, queue.dequeue(), queue.dequeue(), queue.is_empty())

    }
}
