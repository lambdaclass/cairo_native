#![allow(non_snake_case)]

use cairo_felt::Felt252;
use cairo_lang_runner::short_string::as_cairo_short_string;
use std::{fs::File, io::Write, os::fd::FromRawFd};

/// Based on `cairo-lang-runner`'s implementation.
///
/// Source: https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-runner/src/casm_run/mod.rs#L1789-L1800
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
    target_fd: i32,
    data: *const [u8; 32],
    len: usize,
) -> i32 {
    let mut target = File::from_raw_fd(target_fd);

    for i in 0..len {
        let mut data = *data.add(i);
        data.reverse();

        let value = Felt252::from_bytes_be(&data);
        if let Some(shortstring) = as_cairo_short_string(&value) {
            if writeln!(
                target,
                "[DEBUG]\t{shortstring: <31}\t(raw: {})",
                value.to_bigint()
            )
            .is_err()
            {
                return 1;
            };
        } else if writeln!(target, "[DEBUG]\t{:<31}\t(raw: {})", ' ', value.to_bigint()).is_err() {
            return 1;
        }
    }
    if writeln!(target).is_err() {
        return 1;
    };

    0
}
