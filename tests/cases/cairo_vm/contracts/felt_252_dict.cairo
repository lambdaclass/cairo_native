#[contract]
mod Felt252Dict {

use dict::{felt252_dict_entry_finalize, Felt252DictTrait};
    /// An external method that requires the `segment_arena` builtin.
    #[external]
    fn squash_empty_dict() -> bool {
        let x = felt252_dict_new::<felt252>();
        x.squash();
        return true;
    }
}
