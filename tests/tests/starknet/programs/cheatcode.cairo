use core::starknet::testing::cheatcode;

fn set_sequencer_address() {
    let address = 123;
    cheatcode::<'set_sequencer_address'>(array![address].span());
    ()
}
