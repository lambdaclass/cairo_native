// #[feature("bounded-int-utils")]
// use core::internal::bounded_int::{BoundedInt, add, AddHelper, is_zero};

// use core::zeroable::IsZeroResult;
// use core::option::Option;
// // impl AddWeirdBI of AddHelper<BoundedInt<1, 35>, BoundedInt<-2, -1>> {
// //     type Result = BoundedInt<-1, 34>;
// // }

// // #[inline(never)]
// // fn run_test_3(
// //     a: felt252,
// //     b: felt252,
// // ) -> BoundedInt<-1, 34> {
// //     let a: BoundedInt<1, 35> = a.try_into().unwrap();
// //     let b: BoundedInt<-2, -1> = b.try_into().unwrap();
// //     return add(a, b);
// // }

// #[inline(never)]
// fn test_is_zero() -> Option::<core::zeroable::NonZero::<BoundedInt::<-105, 307>>> {
//     let a: BoundedInt<-105, 307> = 0;
//     match is_zero(a) {
//         IsZeroResult::Zero => None,
//         IsZeroResult::NonZero(n) => Some(n),
//     }
// }

// fn main() -> Option::<core::zeroable::NonZero::<BoundedInt::<-105, 307>>> {
//     test_is_zero()
// }
#[feature("bounded-int-utils")]
use core::internal::bounded_int::{self, BoundedInt, constrain, ConstrainHelper, upcast, downcast};

impl ConstrainTest2 of ConstrainHelper<BoundedInt<-10, 10>, 0> {
    type LowT = BoundedInt<-10, -1>;
    type HighT = BoundedInt<0, 10>;
}

fn run_test_5(a: felt252) -> BoundedInt<-10, -1> {
    let a_bi: BoundedInt<-10, 10> = a.try_into().unwrap();
    match constrain::<_, 0>(a_bi) {
        Ok(lt0) => lt0,
        Err(_gt0) => panic!(),
    }
}

fn main() -> BoundedInt<-10, -1> {
    run_test_5(5)
}
