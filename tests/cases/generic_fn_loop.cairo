fn main() -> (u16, u64) {
    (
        bar(3),
        bar(5),
    )
}

fn bar<T, +Drop<T>>(a: T) -> T {
    loop {
        break;
    };
   a
}
