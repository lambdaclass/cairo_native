use ec::ec_point_from_x_nz;

fn main() -> (Option<NonZero<EcPoint>>, Option<NonZero<EcPoint>>) {
    (ec_point_from_x_nz(1234), ec_point_from_x_nz(0), )
}
