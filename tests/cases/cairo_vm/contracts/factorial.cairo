#[contract]
mod Factorial {

    #[external]
    fn factorial(n: felt252) -> felt252 {
        if (n == 0) {
            return 1;
        }
        n * factorial(n - 1)
    }
}
