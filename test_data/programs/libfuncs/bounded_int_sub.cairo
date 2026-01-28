#[feature("bounded-int-utils")]
use core::internal::bounded_int::{BoundedInt, sub, SubHelper};

impl SubHelperBI_1x1_BI_1x5 of SubHelper<BoundedInt<1, 1>, BoundedInt<1, 5>> {
    type Result = BoundedInt<-4, 0>;
}

fn bi_1x1_minus_bi_1x5(
    a: felt252,
    b: felt252,
) -> BoundedInt<-4, 0> {
    let a: BoundedInt<1, 1> = a.try_into().unwrap();
    let b: BoundedInt<1, 5> = b.try_into().unwrap();
    return sub(a, b);
}

impl SubHelperBI_1x1_BI_1x1 of SubHelper<BoundedInt<1, 1>, BoundedInt<1, 1>> {
    type Result = BoundedInt<0, 0>;
}

fn bi_1x1_minus_bi_1x1(
    a: felt252,
    b: felt252,
) -> BoundedInt<0, 0> {
    let a: BoundedInt<1, 1> = a.try_into().unwrap();
    let b: BoundedInt<1, 1> = b.try_into().unwrap();
    return sub(a, b);
}

impl SubHelperBI_m3xm3_BI_m3xm3 of SubHelper<BoundedInt<-3, -3>, BoundedInt<-3, -3>> {
    type Result = BoundedInt<0, 0>;
}

fn bi_m3xm3_minus_bi_m3xm3(
    a: felt252,
    b: felt252,
) -> BoundedInt<0, 0> {
    let a: BoundedInt<-3, -3> = a.try_into().unwrap();
    let b: BoundedInt<-3, -3> = b.try_into().unwrap();
    return sub(a, b);
}

impl SubHelperBI_m6xm3_BI_1x3 of SubHelper<BoundedInt<-6, -3>, BoundedInt<1, 3>> {
    type Result = BoundedInt<-9, -4>;
}

fn bi_m6xm3_minus_bi_1x3(
    a: felt252,
    b: felt252,
) -> BoundedInt<-9, -4> {
    let a: BoundedInt<-6, -3> = a.try_into().unwrap();
    let b: BoundedInt<1, 3> = b.try_into().unwrap();
    return sub(a, b);
}

impl SubHelperBI_m6xm2_BI_m20xm10 of SubHelper<BoundedInt<-6, -2>, BoundedInt<-20, -10>> {
    type Result = BoundedInt<4, 18>;
}

fn bi_m6xm2_minus_bi_m20xm10(
    a: felt252,
    b: felt252,
) -> BoundedInt<4, 18> {
    let a: BoundedInt<-6, -2> = a.try_into().unwrap();
    let b: BoundedInt<-20, -10> = b.try_into().unwrap();
    return sub(a, b);
}
