fn search(array: @Array<felt252>, number: felt252) -> u32 {
    let mut index = 0;

    while index < array.len() {
        if *array[index] == number {
            break;
        }

        index += 1;
    };

    return index;
}

fn init_array(length: u32) -> Array<felt252> {
    let mut array = ArrayTrait::new();
    for i in 0..length {
        array.append(i.into());
    };

    return array;
}

fn main() {
    let array = init_array(4001);

    let index = search(@array, 4000);
    assert(
        index == 400000,
        'invalid result'
    );
    let index = search(@array, 2000);
    assert(
        index == 200000,
        'invalid result'
    );
}
