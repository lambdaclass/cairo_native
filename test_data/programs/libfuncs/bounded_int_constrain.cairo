#[feature("bounded-int-utils")]
use core::internal::bounded_int::{self, BoundedInt, ConstrainHelper, constrain};

fn constrain_bi_m128_127_lt_0(a: felt252) -> BoundedInt<-128, -1> {
    let a: i8 = a.try_into().unwrap();
    match constrain::<i8, 0>(a) {
        Ok(lt0) => lt0,
        Err(_gt0) => panic!(),
    }
}

fn constrain_bi_m128_127_gt_0(a: felt252) -> BoundedInt<0, 127> {
    let a: i8 = a.try_into().unwrap();
    match constrain::<i8, 0>(a) {
        Ok(_lt0) => panic!(),
        Err(gt0) => gt0,
    }
}

impl ConstrainTest1 of ConstrainHelper<BoundedInt<0, 15>, 5> {
    type LowT = BoundedInt<0, 4>;
    type HighT = BoundedInt<5, 15>;
}

fn constrain_bi_0_15_lt_5(a: felt252) -> BoundedInt<0, 4> {
    let a_bi: BoundedInt<0, 15> = a.try_into().unwrap();
    match constrain::<_, 5>(a_bi) {
        Ok(lt) => lt,
        Err(_gt) => panic!(),
    }
}

fn constrain_bi_0_15_gt_5(a: felt252) -> BoundedInt<5, 15> {
    let a_bi: BoundedInt<0, 15> = a.try_into().unwrap();
    match constrain::<_, 5>(a_bi) {
        Ok(_lt) => panic!(),
        Err(gt) => gt,
    }
}

impl ConstrainTest2 of ConstrainHelper<BoundedInt<-10, 10>, 0> {
    type LowT = BoundedInt<-10, -1>;
    type HighT = BoundedInt<0, 10>;
}

fn constrain_bi_m10_10_lt_0(a: felt252) -> BoundedInt<-10, -1> {
    let a_bi: BoundedInt<-10, 10> = a.try_into().unwrap();
    match constrain::<_, 0>(a_bi) {
        Ok(lt0) => lt0,
        Err(_gt0) => panic!(),
    }
}

fn constrain_bi_m10_10_gt_0(a: felt252) -> BoundedInt<0, 10> {
    let a_bi: BoundedInt<-10, 10> = a.try_into().unwrap();
    match constrain::<_, 0>(a_bi) {
        Ok(_lt0) => panic!(),
        Err(gt0) => gt0,
    }
}

impl ConstrainTest3 of ConstrainHelper<BoundedInt<1, 61>, 31> {
    type LowT = BoundedInt<1, 30>;
    type HighT = BoundedInt<31, 61>;
}

fn constrain_bi_1_61_lt_31(a: felt252) -> BoundedInt<1, 30> {
    let a_bi: BoundedInt<1, 61> = a.try_into().unwrap();
    match constrain::<_, 31>(a_bi) {
        Ok(lt) => lt,
        Err(_gt) => panic!(),
    }
}

fn constrain_bi_1_61_gt_31(a: felt252) -> BoundedInt<31, 61> {
    let a_bi: BoundedInt<1, 61> = a.try_into().unwrap();
    match constrain::<_, 31>(a_bi) {
        Ok(_lt) => panic!(),
        Err(gt) => gt,
    }
}

impl ConstrainTest4 of ConstrainHelper<BoundedInt<-200, -100>, -150> {
    type LowT = BoundedInt<-200, -151>;
    type HighT = BoundedInt<-150, -100>;
}

fn constrain_bi_m200_m100_lt_m150(a: felt252) -> BoundedInt<-200, -151> {
    let a_bi: BoundedInt<-200, -100> = a.try_into().unwrap();
    match constrain::<_, -150>(a_bi) {
        Ok(lt) => lt,
        Err(_gt) => panic!(),
    }
}

fn constrain_bi_m200_m100_gt_m150(a: felt252) -> BoundedInt<-150, -100> {
    let a_bi: BoundedInt<-200, -100> = a.try_into().unwrap();
    match constrain::<_, -150>(a_bi) {
        Ok(_lt) => panic!(),
        Err(gt) => gt,
    }
}

impl ConstrainTest5 of ConstrainHelper<BoundedInt<30, 100>, 100> {
    type LowT = BoundedInt<30, 99>;
    type HighT = BoundedInt<100, 100>;
}

fn constrain_bi_30_100_gt_100(a: felt252) -> BoundedInt<100, 100> {
    let a_bi: BoundedInt<30, 100> = a.try_into().unwrap();
    match constrain::<_, 100>(a_bi) {
        Ok(_lt) => panic!(),
        Err(gt) => gt,
    }
}

impl ConstrainTest6 of ConstrainHelper<BoundedInt<-30, 31>, 0> {
    type LowT = BoundedInt<-30, -1>;
    type HighT = BoundedInt<0, 31>;
}

fn constrain_bi_m30_31_lt_0(a: felt252) -> BoundedInt<-30, -1> {
    let a_bi: BoundedInt<-30, 31> = a.try_into().unwrap();
    match constrain::<_, 0>(a_bi) {
        Ok(lt0) => lt0,
        Err(_gt0) => panic!(),
    }
}

fn constrain_bi_m30_31_gt_0(a: felt252) -> BoundedInt<0, 31> {
    let a_bi: BoundedInt<-30, 31> = a.try_into().unwrap();
    match constrain::<_, 0>(a_bi) {
        Ok(_lt0) => panic!(),
        Err(gt0) => gt0,
    }
}
