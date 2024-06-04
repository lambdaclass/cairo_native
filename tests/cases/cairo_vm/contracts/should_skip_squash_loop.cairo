#[contract]
mod ShouldSkipSquashLoop {
    use dict::Felt252DictTrait;

    #[external]
    fn should_skip_squash_loop() {
        let x = felt252_dict_new::<felt252>();
        x.squash();
    }
}
