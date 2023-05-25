// func main() {
//     fib(1, 1, 1500000);
// 
//     ret;
// }
// 
// func fib(first_element, second_element, n) -> (res: felt) {
//     if (n == 0) {
//         return (second_element,);
//     }
// 
//     tempvar y = first_element + second_element;
//     return fib(second_element, y, n - 1);
// }

fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
    match n {
        0 => a,
        _ => fib(b, a + b, n - 1),
    }
}

#[test]
fn main() -> felt252 {
    let y: felt252 = fib(1, 1, 15000);
}

