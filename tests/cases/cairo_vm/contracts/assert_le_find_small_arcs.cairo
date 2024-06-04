#[contract]
  mod Felt252Dict {

  use dict::{felt252_dict_entry_finalize, Felt252DictTrait};

      #[external]
      fn update_dict() {
        let mut dict = felt252_dict_new::<felt252>();
        dict.insert(1, 64);
        dict.insert(2, 75);
        dict.insert(3, 75);
        dict.squash();
     }
 }
