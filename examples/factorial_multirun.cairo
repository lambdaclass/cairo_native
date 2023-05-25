// // factorial(n) =  n!
// func factorial(n) -> (result: felt) {
//     if (n == 1) {
//         return (n,);
//     }
//     let (a) = factorial(n - 1);
//     return (n * a,);
// }
//
// func main() {
//     // Make sure the factorial(10) == 3628800
//     let (y) = factorial(10);
//     y = 3628800;
//
//     factorial(2000000);
//     return ();
// }

use debug::PrintTrait;

fn factorial(n: felt252) -> felt252 {
    if (n == 1) {
        n
    } else {
        n * factorial(n - 1)
    }
}

#[test]
fn main() {
    // Make sure that factorial(10) == 3628800
    let y: felt252 = factorial(10);
    assert(3628800 == y, 'failed test');
    factorial(10000).print();
}
