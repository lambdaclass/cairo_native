#[contract]
mod SegmentArenaIndex {
use dict::Felt252DictTrait;

    #[external]
    fn test_arena_index() -> bool {
        let mut dict: Felt252Dict<felt252> = Felt252DictTrait::new();
        let squashed_dict = dict.squash();
        return true;
    }
}
