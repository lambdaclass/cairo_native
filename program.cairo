 enum MyEnum {
     A: felt252,
     B: u8,
     C: u16,
     D: u32,
     E: u64,
 }

  fn match_b() -> u8 {
      let x = MyEnum::B(5_u8);
      match x {
          MyEnum::A(_) => 0_u8,
          MyEnum::B(x) => x,
          MyEnum::C(_) => 1_u8,
          MyEnum::D(_) => 2_u8,
          MyEnum::E(_) => 3_u8,
      }
  }
