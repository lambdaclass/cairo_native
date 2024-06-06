#[starknet::contract]
mod ShouldSkipSquashLoop {
    #[storage]
    struct Storage {}

    use dict::Felt252DictTrait;

    #[external(v0)]
    fn should_skip_squash_loop(self: @ContractState) {
        let x = felt252_dict_new::<felt252>();
        x.squash();
    }
}
