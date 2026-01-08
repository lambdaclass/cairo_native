fn fibonacci(n: felt252) -> felt252 {
    if (n == 0 || n == 1) {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

#[cfg(test)]
mod test {
    use super::fibonacci;

    #[test]
    fn fibonacci_1() {
        assert_eq!(fibonacci(1), 1);
    }

    #[test]
    fn fibonacci_2() {
        assert_eq!(fibonacci(2), 1);
    }

    #[test]
    fn fibonacci_3() {
        assert_eq!(fibonacci(3), 2);
    }

    #[test]
    fn fibonacci_4() {
        assert_eq!(fibonacci(4), 3);
    }
}
