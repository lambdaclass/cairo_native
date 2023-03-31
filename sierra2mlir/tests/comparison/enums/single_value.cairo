enum Single {
	A: felt252
}

fn main() -> felt252 {
	let s = Single::A(5);
	match s {
		Single::A(x) => x
	}
}