use traits::Default;
 use dict::Felt252DictTrait;
 fn run_test() -> u32 {
     let mut dict: Felt252Dict<u32> = Default::default();
     dict.insert(2, 1_u32);
     dict.insert(3, 1_u32);
     dict.insert(4, 1_u32);
     dict.insert(5, 1_u32);
     dict.insert(6, 1_u32);
     dict.insert(7, 1_u32);
     dict.insert(8, 1_u32);
     dict.insert(9, 1_u32);
     dict.insert(10, 1_u32);
     dict.insert(11, 1_u32);
     dict.insert(12, 1_u32);
     dict.insert(13, 1_u32);
     dict.insert(14, 1_u32);
     dict.insert(15, 1_u32);
     dict.insert(16, 1_u32);
     dict.insert(17, 1_u32);
     dict.insert(18, 1345432_u32);
     dict.get(18)
 }
