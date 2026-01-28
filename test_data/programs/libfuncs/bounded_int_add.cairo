#[feature("bounded-int-utils")]
use core::internal::bounded_int::{BoundedInt, add, AddHelper, UnitInt};

impl AddHelperBI_1x31_BI_1x1 of AddHelper<BoundedInt<1, 31>, BoundedInt<1, 1>> {
    type Result = BoundedInt<2, 32>;
}

fn bi_1x31_plus_bi_1x1(
    a: felt252,
    b: felt252,
) -> BoundedInt<2, 32> {
    let a: BoundedInt<1, 31> = a.try_into().unwrap();
    let b: BoundedInt<1, 1> = b.try_into().unwrap();
    return add(a, b);
}

impl AddHelperBI_1x31_BI_m1xm1 of AddHelper<BoundedInt<1, 31>, BoundedInt<-1, -1>> {
    type Result = BoundedInt<0, 30>;
}

fn bi_1x31_plus_bi_m1xm1(
    a: felt252,
    b: felt252,
) -> BoundedInt<0, 30> {
    let a: BoundedInt<1, 31> = a.try_into().unwrap();
    let b: BoundedInt<-1, -1> = b.try_into().unwrap();
    return add(a, b);
}

impl AddHelperBI_0x30_BI_0x10 of AddHelper<BoundedInt<0, 30>, BoundedInt<0, 10>> {
    type Result = BoundedInt<0, 40>;
}

fn bi_0x30_plus_bi_0x10(
    a: felt252,
    b: felt252,
) -> BoundedInt<0, 40> {
    let a: BoundedInt<0, 30> = a.try_into().unwrap();
    let b: BoundedInt<0, 10> = b.try_into().unwrap();
    return add(a, b);
}

impl AddHelperBI_m20xm15_BI_0x10 of AddHelper<BoundedInt<-20, -15>, BoundedInt<0, 10>> {
    type Result = BoundedInt<-20, -5>;
}

fn bi_m20xm15_plus_bi_0x10(
    a: felt252,
    b: felt252,
) -> BoundedInt<-20, -5> {
    let a: BoundedInt<-20, -15> = a.try_into().unwrap();
    let b: BoundedInt<0, 10> = b.try_into().unwrap();
    return add(a, b);
}

impl AddHelperBI_m5xm5_BI_m5xm5 of AddHelper<BoundedInt<-5, -5>, BoundedInt<-5, -5>> {
    type Result = BoundedInt<-10, -10>;
}

fn bi_m5xm5_plus_bi_m5xm5(
    a: felt252,
    b: felt252,
) -> BoundedInt<-10, -10> {
    let a: BoundedInt<-5, -5> = a.try_into().unwrap();
    let b: BoundedInt<-5, -5> = b.try_into().unwrap();
    return add(a, b);
}

impl AddHelperBI_m5xm5_UI_m1 of AddHelper<BoundedInt<-5, -5>, UnitInt<-1>> {
    type Result = BoundedInt<-6, -6>;
}

fn bi_m5xm5_plus_ui_m1(
    a: felt252,
    b: felt252,
) -> BoundedInt<-6, -6> {
    let a: BoundedInt<-5, -5> = a.try_into().unwrap();
    let b: UnitInt<-1> = b.try_into().unwrap();
    return add(a, b);
}

impl AddHelperUI_1_BI_m5xm5 of AddHelper<UnitInt<1>, BoundedInt<-5, -5>> {
    type Result = BoundedInt<-4, -4>;
}

fn ui_m1_plus_bi_m5xm5(
    a: felt252,
    b: felt252,
) -> BoundedInt<-4, -4> {
    let a: UnitInt<1> = a.try_into().unwrap();
    let b: BoundedInt<-5, -5> = b.try_into().unwrap();
    return add(a, b);
}
