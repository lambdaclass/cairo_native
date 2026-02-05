use zeroable::IsZeroResult;
use zeroable::NonZeroIntoImpl;

fn main() -> felt252 {
    NonZeroIntoImpl::into(felt_to_nonzero(1234))
}

fn felt_to_nonzero(value: felt252) -> NonZero<felt252> {
    match felt252_is_zero(value) {
        IsZeroResult::Zero(()) => panic(ArrayTrait::new()),
        IsZeroResult::NonZero(x) => x,
    }
}
