use ec::ec_point_try_new_nz;

fn main() -> (
    Option<NonZero<EcPoint>>,
    Option<NonZero<EcPoint>>,
    Option<NonZero<EcPoint>>,
    Option<NonZero<EcPoint>>,
) {
    (
        ec_point_try_new_nz(
            1234, -1301976514684871091717790968549291947487646995000837413367950573852273027507
        ),
        ec_point_try_new_nz(
            1234, 1301976514684871091717790968549291947487646995000837413367950573852273027507
        ),
        ec_point_try_new_nz(1234, 0),
        ec_point_try_new_nz(1234, 1234),
    )
}
