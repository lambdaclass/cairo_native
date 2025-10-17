#[starknet::contract]
mod HeavyBlake2s {
    use core::blake::{blake2s_finalize, blake2s_compress};
    
    #[storage]
    struct Storage {}

    #[external(v0)]
    fn heavy_blake2s(self: @ContractState) -> [u32; 8] {
        let initial_state = BoxTrait::new([0x13245678, 0x87654321, 0x21324354, 0x54433221, 0x11223344, 0x44332211, 0x55667788, 0x88776655]);
        let msg = BoxTrait::new(['Hi', ',', 'this', 'is', 'a', 'msg', 'to', 'send', 0, 0, 0, 0, 0, 0, 0, 0]);
        let byte_count = 4;
        let mut counter = 0;

        while counter != 6 {
            let _res = blake2s_compress(initial_state, byte_count, msg);
            counter += 1;
        }
        
        blake2s_finalize(initial_state, byte_count, msg).unbox()
    }
}
