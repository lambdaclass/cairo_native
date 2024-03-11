use core::{
    integer,
    integer::{
        BoundedInt, u128_sqrt, u128_wrapping_sub, u16_sqrt, u256_sqrt, u256_wide_mul, u32_sqrt,
        u512_safe_div_rem_by_u256, u512, u64_sqrt, u8_sqrt
    }
};
use core::test::test_utils::{assert_eq, assert_ne, assert_le, assert_lt, assert_gt, assert_ge};

mod special_casts {
    extern type BoundedInt<const MIN: felt252, const MAX: felt252>;
    extern fn downcast<T, S>(index: T) -> Option<S> implicits(RangeCheck) nopanic;
    extern fn upcast<T, S>(index: T) -> S nopanic;

    impl DropBoundedInt120_180 of Drop<BoundedInt<120, 180>>;
    const U128_UPPER: felt252 = 0x100000000000000000000000000000000;
    type BoundedIntU128Upper =
        BoundedInt<0x100000000000000000000000000000000, 0x100000000000000000000000000000000>;
    const U128_MAX: felt252 = 0xffffffffffffffffffffffffffffffff;
    type BoundedIntU128Max =
        BoundedInt<0xffffffffffffffffffffffffffffffff, 0xffffffffffffffffffffffffffffffff>;

    /// Is `value` the equivalent value of `expected` in `T` type.
    fn is_some_of<T>(value: Option<T>, expected: felt252) -> bool {
        match value {
            Option::Some(v) => upcast(v) == expected,
            Option::None => false,
        }
    }

    /// Is `value` the equivalent value (as `felt252`) of `expected` in `T` type.
    fn felt252_downcast_valid<T>(value: felt252) -> bool {
        is_some_of(downcast::<felt252, T>(value), value)
    }

    /// Is `value` the equivalent value (as `felt252`) of `expected` in `T` type.
    fn downcast_invalid<T, S>(value: T) -> bool {
        match downcast::<T, S>(value) {
            Option::Some(v) => {
                // Just as a drop for `v`.
                upcast::<_, felt252>(v);
                false
            },
            Option::None => true,
        }
    }

    #[test]
    fn test_felt252_downcasts() {
        assert!(downcast_invalid::<felt252, BoundedInt<0, 0>>(1));
        assert!(felt252_downcast_valid::<BoundedInt<0, 0>>(0));
        assert!(downcast_invalid::<felt252, BoundedInt<0, 0>>(-1));
        assert!(downcast_invalid::<felt252, BoundedInt<-1, -1>>(-2));
        assert!(felt252_downcast_valid::<BoundedInt<-1, -1>>(-1));
         assert!(downcast_invalid::<felt252, BoundedInt<-1, -1>>(0));
         assert!(downcast_invalid::<felt252, BoundedInt<120, 180>>(119));
         assert!(felt252_downcast_valid::<BoundedInt<120, 180>>(120));
         assert!(felt252_downcast_valid::<BoundedInt<120, 180>>(180));
         assert!(downcast_invalid::<felt252, BoundedInt<120, 180>>(181));
         assert!(downcast_invalid::<felt252, BoundedIntU128Max>(U128_MAX - 1));
         assert!(felt252_downcast_valid::<BoundedIntU128Max>(U128_MAX));
         assert!(downcast_invalid::<felt252, BoundedIntU128Max>(U128_MAX + 1));
         assert!(downcast_invalid::<felt252, BoundedIntU128Upper>(U128_UPPER - 1));
         assert!(felt252_downcast_valid::<BoundedIntU128Upper>(U128_UPPER));
         assert!(downcast_invalid::<felt252, BoundedIntU128Upper>(U128_UPPER + 1));
    }

    // Full prime range, but where the max element is 0.
    type OneMinusPToZero =
        BoundedInt<-0x800000000000011000000000000000000000000000000000000000000000000, 0>;

    type OneMinusPOnly =
        BoundedInt<
            -0x800000000000011000000000000000000000000000000000000000000000000,
            -0x800000000000011000000000000000000000000000000000000000000000000
        >;

    // TODO: fix and enable
    // #[test]
    // fn test_bounded_int_casts() {
    //     let minus_1 = downcast::<felt252, BoundedInt<-1, -1>>(-1).unwrap();
    //     assert!(downcast::<OneMinusPToZero, u8>(upcast(minus_1)).is_none());
    //     let zero = downcast::<felt252, BoundedInt<0, 0>>(0).unwrap();
    //     assert!(downcast::<OneMinusPToZero, u8>(upcast(zero)) == Option::Some(0));
    //     let one_minus_p = downcast::<felt252, OneMinusPOnly>(1).unwrap();
    //     assert!(downcast::<OneMinusPToZero, u8>(upcast(one_minus_p)).is_none());
    //     let v119 = downcast::<felt252, BoundedInt<119, 119>>(119).unwrap();
    //     assert!(downcast::<BoundedInt<100, 200>, BoundedInt<120, 180>>(upcast(v119)).is_none());
    //     let v120 = downcast::<felt252, BoundedInt<120, 120>>(120).unwrap();
    //     assert!(
    //         is_some_of(downcast::<BoundedInt<100, 200>, BoundedInt<120, 180>>(upcast(v120)), 120)
    //     );
    //     let v180 = downcast::<felt252, BoundedInt<180, 180>>(180).unwrap();
    //     assert!(
    //         is_some_of(downcast::<BoundedInt<100, 200>, BoundedInt<120, 180>>(upcast(v180)), 180)
    //     );
    //     let v181 = downcast::<felt252, BoundedInt<181, 181>>(181).unwrap();
    //     assert!(downcast::<BoundedInt<100, 200>, BoundedInt<120, 180>>(upcast(v181)).is_none());
    // }

}
