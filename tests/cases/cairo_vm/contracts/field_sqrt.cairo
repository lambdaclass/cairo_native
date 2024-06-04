#[contract]
mod FieldSqrt {
    use core::traits::Into;
    use option::OptionTrait;   
    use ec::ec_point_from_x;
    use ec::ec_point_non_zero;
    use ec::ec_point_unwrap;

    #[external]
    fn field_sqrt() -> felt252 {
        let p = ec_point_from_x(10).unwrap();
        let p_nz = ec_point_non_zero(p);

        let (qx, _) = ec_point_unwrap(p_nz);

        assert(qx == 10, 'bad finalize x');
        qx
    }
}
