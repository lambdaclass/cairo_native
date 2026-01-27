use core::box::BoxTrait;

struct Hello {
    x: i32,
}

fn run_test() -> Hello {
    let x = BoxTrait::new(Hello {
        x: -2
    });
    x.unbox()
}
