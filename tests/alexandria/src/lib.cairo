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
}
