#[derive(Drop)]
struct Bye {
    a: Hello
}

#[derive(Drop)]
struct Hello {
    a: Empty
}

#[derive(Drop)]
struct Empty {}

trait Hi {
    fn hi(self: @Empty);
    fn bye(self: @Empty);
    fn hello(self: @Empty);
}

impl Himpl of Hi {
    fn hi(self: @Empty) {
        println!("hi");
    }
    fn bye(self: @Empty) {
        println!("bye");
    }
    fn hello(self: @Empty) {
        println!("hello");
    }
}

fn main() {
    let strct = Bye {
        a: Hello {
            a: Empty {}
        }
    };

    strct.a.a.hi();
    strct.a.a.bye();
    strct.a.a.hello();
}
