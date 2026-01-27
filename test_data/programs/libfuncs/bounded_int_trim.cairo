#[feature("bounded-int-utils")]
use core::internal::bounded_int::{self, BoundedInt, trim_min, trim_max, TrimMinHelper, TrimMaxHelper};
use core::internal::OptionRev;


fn test_i8_min(a: felt252) {
    let a_int: i8 = a.try_into().unwrap();
    match trim_min::<i8>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}
fn test_i8_max(a: felt252) {
    let a_int: i8 = a.try_into().unwrap();
    match trim_max::<i8>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

fn test_u8_min(a: felt252) {
    let a_int: u8 = a.try_into().unwrap();
    match trim_min::<u8>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}
fn test_u8_max(a: felt252) {
    let a_int: u8 = a.try_into().unwrap();
    match trim_max::<u8>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MinHelper_0_100 of TrimMinHelper<BoundedInt<0, 100>> {
    type Target = BoundedInt<1, 100>;
}
fn test_0_100_min(a: felt252) {
    let a_int: BoundedInt<0, 100> = a.try_into().unwrap();
    match trim_min::<BoundedInt<0, 100>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MaxHelper_0_100 of TrimMaxHelper<BoundedInt<0, 100>> {
    type Target = BoundedInt<0, 99>;
}
fn test_0_100_max(a: felt252) {
    let a_int: BoundedInt<0, 100> = a.try_into().unwrap();
    match trim_max::<BoundedInt<0, 100>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MinHelper_10_100 of TrimMinHelper<BoundedInt<10, 100>> {
    type Target = BoundedInt<11, 100>;
}
fn test_10_100_min(a: felt252) {
    let a_int: BoundedInt<10, 100> = a.try_into().unwrap();
    match trim_min::<BoundedInt<10, 100>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MaxHelper_10_100 of TrimMaxHelper<BoundedInt<10, 100>> {
    type Target = BoundedInt<10, 99>;
}
fn test_10_100_max(a: felt252) {
    let a_int: BoundedInt<10, 100> = a.try_into().unwrap();
    match trim_max::<BoundedInt<10, 100>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MinHelper_m100_0 of TrimMinHelper<BoundedInt<-100, 0>> {
    type Target = BoundedInt<-99, 0>;
}
fn test_m100_0_min(a: felt252) {
    let a_int: BoundedInt<-100, 0> = a.try_into().unwrap();
    match trim_min::<BoundedInt<-100, 0>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MaxHelper_m100_0 of TrimMaxHelper<BoundedInt<-100, 0>> {
    type Target = BoundedInt<-100, -1>;
}
fn test_m100_0_max(a: felt252) {
    let a_int: BoundedInt<-100, 0> = a.try_into().unwrap();
    match trim_max::<BoundedInt<-100, 0>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MinHelper_m100_m10 of TrimMinHelper<BoundedInt<-100, -10>> {
    type Target = BoundedInt<-99, -10>;
}
fn test_m100_m10_min(a: felt252) {
    let a_int: BoundedInt<-100, -10> = a.try_into().unwrap();
    match trim_min::<BoundedInt<-100, -10>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MaxHelper_m100_m10 of TrimMaxHelper<BoundedInt<-100, -10>> {
    type Target = BoundedInt<-100, -11>;
}
fn test_m100_m10_max(a: felt252) {
    let a_int: BoundedInt<-100, -10> = a.try_into().unwrap();
    match trim_max::<BoundedInt<-100, -10>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MinHelper_m100_100 of TrimMinHelper<BoundedInt<-100, 100>> {
    type Target = BoundedInt<-99, 100>;
}
fn test_m100_100_min(a: felt252) {
    let a_int: BoundedInt<-100, 100> = a.try_into().unwrap();
    match trim_min::<BoundedInt<-100, 100>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MaxHelper_m100_100 of TrimMaxHelper<BoundedInt<-100, 100>> {
    type Target = BoundedInt<-100, 99>;
}
fn test_m100_100_max(a: felt252) {
    let a_int: BoundedInt<-100, 100> = a.try_into().unwrap();
    match trim_max::<BoundedInt<-100, 100>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}

impl MinHelper_0_8 of TrimMinHelper<BoundedInt<0, 8>> {
    type Target = BoundedInt<1, 8>;
}
fn test_0_8_min(a: felt252) {
    let a_int: BoundedInt<0, 8> = a.try_into().unwrap();
    match trim_min::<BoundedInt<0, 8>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}
impl MaxHelper_0_8 of TrimMaxHelper<BoundedInt<0, 8>> {
    type Target = BoundedInt<0, 7>;
}
fn test_0_8_max(a: felt252) {
    let a_int: BoundedInt<0, 8> = a.try_into().unwrap();
    match trim_max::<BoundedInt<0, 8>>(a_int) {
        OptionRev::Some(v) => assert!(v == a.try_into().unwrap(), "invariant"),
        OptionRev::None => panic!("boundary"),
    };
}
