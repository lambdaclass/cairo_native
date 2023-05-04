use ec::ec_neg;
use ec::ec_point_from_x_nz;
use ec::ec_point_zero;
use option::OptionTrait;
use zeroable::NonZeroIntoImpl;

fn main() -> (EcPoint, EcPoint) {
    (
        ec_neg(ec_point_zero()),
        ec_neg(NonZeroIntoImpl::into(ec_point_from_x_nz(1234).unwrap())),
    )
}
