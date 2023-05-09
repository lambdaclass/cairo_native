use ec::IsZeroResult;
use ec::ec_point_from_x_nz;
use ec::ec_point_is_zero;
use ec::ec_point_zero;
use option::OptionTraitImpl;
use zeroable::unwrap_non_zero;

fn main() -> (IsZeroResult<EcPoint>, IsZeroResult<EcPoint>) {
    (
        ec_point_is_zero(unwrap_non_zero(OptionTraitImpl::unwrap(ec_point_from_x_nz(1234)))),
        ec_point_is_zero(ec_point_zero()),
    )
}
