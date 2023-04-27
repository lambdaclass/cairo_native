use ec::ec_point_from_x_nz;
use ec::ec_point_unwrap;
use option::OptionTrait;

fn main() -> (felt252, felt252) {
    ec_point_unwrap(ec_point_from_x_nz(1234).unwrap())
}
