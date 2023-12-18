mod alexandria {
    fn fib() -> felt252 {
        alexandria_math::fibonacci::fib(16, 10, 1)
    }

    fn karatsuba() -> u128 {
        alexandria_math::karatsuba::multiply(3754192357923759273591, 18492875)
    }
}
