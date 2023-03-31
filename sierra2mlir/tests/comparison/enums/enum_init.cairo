enum MyEnum {
	A: felt252,
	B: (felt252, felt252),
}

fn main() -> (MyEnum, MyEnum) {
	(MyEnum::A(10), MyEnum::B((20, 30)))
}
