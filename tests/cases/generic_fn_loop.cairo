fn main() -> (u16, u64) {
    (
        bar(3),
        bar(5),
    )
}

fn bar<T, +Drop<T>>(a: T) -> T {
   let mut i: usize = 0;
    loop {
        if i == 10 {
            break;
        }
        i += 1;
    };
   a
}
