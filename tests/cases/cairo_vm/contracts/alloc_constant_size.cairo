#[contract]
mod AllocConstantSizeContract {
use box::BoxTrait;
// Calculates fib, but all variables are boxes.
    #[external]
    fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
        let a = BoxTrait::new(a);
        let b = BoxTrait::new(b);
        let n = BoxTrait::new(n);


        let unboxed_n = n.unbox(); 
        if unboxed_n == 0 {
            a.unbox()
        } else {
            fib(b.unbox(), a.unbox() + b.unbox(), unboxed_n - 1, )
        }
    }
}
