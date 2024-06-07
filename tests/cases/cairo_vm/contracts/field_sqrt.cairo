#[starknet::contract]
mod FieldSqrt {
    #[storage]
    struct Storage {}

    use core::traits::Into;
    use option::OptionTrait;   
    use ec::ec_point_from_x_nz;
    use ec::ec_point_unwrap;

    #[external(v0)]
    fn field_sqrt(self: @ContractState) -> felt252 {
        let p_nz = ec_point_from_x_nz(10).unwrap();

        let (qx, _) = ec_point_unwrap(p_nz);

        assert(qx == 10, 'bad finalize x');
        qx
    }
}
