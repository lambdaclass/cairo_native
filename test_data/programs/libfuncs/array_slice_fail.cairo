use array::Array;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use box::BoxTrait;

fn run_test() -> u32 {
    let mut data: Array<u32> = ArrayTrait::new();
    data.append(1_u32);
    data.append(2_u32);
    data.append(3_u32);
    data.append(4_u32);
    let sp = data.span();
    let slice = sp.slice(1, 4); // oob
    //data.append(5_u32);
    *slice.get(0).unwrap().unbox()
}
