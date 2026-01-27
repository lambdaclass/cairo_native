use array::Array;
use array::ArrayTrait;

fn main() {
    let mut data: Array<felt252> = ArrayTrait::new();
    data.append('This is an error');
    data.append('Spanning over two felts');
    panic(data);
}
