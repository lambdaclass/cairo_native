#![allow(non_snake_case)]

use cairo_lang_sierra_gas::core_libfunc_cost::{
    DICT_SQUASH_REPEATED_ACCESS_COST, DICT_SQUASH_UNIQUE_KEY_COST,
};
use lazy_static::lazy_static;
use rand::Rng;
use starknet_curve::curve_params::BETA;
use starknet_types_core::{
    curve::{AffinePoint, ProjectivePoint},
    felt::Felt,
};
use std::ops::Mul;
use std::{collections::HashMap, fs::File, io::Write, os::fd::FromRawFd, ptr::NonNull, slice};

lazy_static! {
    pub static ref HALF_PRIME: Felt = Felt::from_dec_str(
        "1809251394333065606848661391547535052811553607665798349986546028067936010240"
    )
    .unwrap();
    pub static ref DICT_GAS_REFUND_PER_ACCESS: u64 =
        (DICT_SQUASH_UNIQUE_KEY_COST.cost() - DICT_SQUASH_REPEATED_ACCESS_COST.cost()) as u64;
}

/// Based on `cairo-lang-runner`'s implementation.
///
/// Source: <https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-runner/src/casm_run/mod.rs#L1946-L1948>
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
    target_fd: i32,
    data: *const [u8; 32],
    len: u32,
) -> i32 {
    let mut target = File::from_raw_fd(target_fd);

    for i in 0..len as usize {
        let data = *data.add(i);

        let value = Felt::from_bytes_le(&data);
        if write!(target, "[DEBUG]\t0x{value:x}",).is_err() {
            return 1;
        };

        if data[..32]
            .iter()
            .copied()
            .all(|ch| ch == 0 || ch.is_ascii_graphic() || ch.is_ascii_whitespace())
        {
            let mut buf = [0; 31];
            let mut len = 31;
            for &ch in data.iter().take(31) {
                if ch != 0 {
                    len -= 1;
                    buf[len] = ch;
                }
            }

            if write!(
                target,
                " ('{}')",
                std::str::from_utf8_unchecked(&buf[len..])
            )
            .is_err()
            {
                return 1;
            }
        }

        if writeln!(target).is_err() {
            return 1;
        };
    }

    // Avoid closing `stdout`.
    std::mem::forget(target);

    0
}

/// Compute `pedersen(lhs, rhs)` and store it into `dst`.
///
/// All its operands need the values in big endian.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__pedersen(
    dst: *mut u8,
    lhs: *const u8,
    rhs: *const u8,
) {
    // Extract arrays from the pointers.
    let dst = slice::from_raw_parts_mut(dst, 32);
    let lhs = slice::from_raw_parts(lhs, 32);
    let rhs = slice::from_raw_parts(rhs, 32);

    // Convert to FieldElement.
    let lhs = Felt::from_bytes_le_slice(lhs);
    let rhs = Felt::from_bytes_le_slice(rhs);

    // Compute pedersen hash and copy the result into `dst`.
    let res = starknet_crypto::pedersen_hash(&lhs, &rhs);
    dst.copy_from_slice(&res.to_bytes_le());
}

/// Compute `hades_permutation(op0, op1, op2)` and replace the operands with the results.
///
/// All operands need the values in big endian.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__hades_permutation(
    op0: *mut u8,
    op1: *mut u8,
    op2: *mut u8,
) {
    // Extract arrays from the pointers.
    let op0 = slice::from_raw_parts_mut(op0, 32);
    let op1 = slice::from_raw_parts_mut(op1, 32);
    let op2 = slice::from_raw_parts_mut(op2, 32);

    // Convert to FieldElement.
    let mut state = [
        Felt::from_bytes_le_slice(op0),
        Felt::from_bytes_le_slice(op1),
        Felt::from_bytes_le_slice(op2),
    ];

    // Compute Poseidon permutation.
    starknet_crypto::poseidon_permute_comp(&mut state);

    // Write back the results.
    op0.copy_from_slice(&state[0].to_bytes_le());
    op1.copy_from_slice(&state[1].to_bytes_le());
    op2.copy_from_slice(&state[2].to_bytes_le());
}

/// Felt252 type used in cairo native runtime
pub type FeltDict = (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64);

/// Allocates a new dictionary. Internally a rust hashmap: `HashMap<[u8; 32], NonNull<()>`
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__alloc_dict() -> *mut std::ffi::c_void {
    Box::into_raw(Box::<FeltDict>::default()) as _
}

/// Frees the dictionary.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_free(ptr: *mut FeltDict) {
    let mut map = Box::from_raw(ptr);

    // Free the entries manually.
    for (_, entry) in map.as_mut().0.drain() {
        libc::free(entry.as_ptr().cast());
    }
}

/// Needed for the correct alignment,
/// since the key [u8; 32] in rust has 8 byte alignment but its a felt,
/// so in reality it has 16.
#[repr(C, align(16))]
pub struct DictValuesArrayAbi {
    pub key: [u8; 32],
    pub value: std::ptr::NonNull<libc::c_void>,
}

/// Returns a array over the values of the dict, used for deep cloning.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_values(
    ptr: *mut FeltDict,
    len: *mut u64,
) -> *mut DictValuesArrayAbi {
    let dict: &mut FeltDict = &mut *ptr;

    let values: Vec<_> = dict
        .0
        .clone()
        .into_iter()
        // make it ffi safe for use within MLIR.
        .map(|x| DictValuesArrayAbi {
            key: x.0,
            value: x.1,
        })
        .collect();
    *len = values.len() as u64;
    values.leak::<'static>().as_mut_ptr()
}

/// Gets the value for a given key, the returned pointer is null if not found.
/// Increments the access count.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_get(
    ptr: *mut FeltDict,
    key: &[u8; 32],
) -> *mut std::ffi::c_void {
    let dict: &mut FeltDict = &mut *ptr;
    let map = &dict.0;
    dict.1 += 1;

    if let Some(v) = map.get(key) {
        v.as_ptr()
    } else {
        std::ptr::null_mut()
    }
}

/// Inserts the provided key value. Returning the old one or nullptr if there was none.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_insert(
    ptr: *mut FeltDict,
    key: &[u8; 32],
    value: NonNull<std::ffi::c_void>,
) -> *mut std::ffi::c_void {
    let dict = &mut *ptr;
    let old_ptr = dict.0.insert(*key, value);

    if let Some(v) = old_ptr {
        v.as_ptr()
    } else {
        std::ptr::null_mut()
    }
}

/// Compute the total gas refund for the dictionary at squash time.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_gas_refund(ptr: *const FeltDict) -> u64 {
    let dict = &*ptr;
    (dict.1 - dict.0.len() as u64) * *DICT_GAS_REFUND_PER_ACCESS
}

/// Compute `ec_point_from_x_nz(x)` and store it.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_from_x_nz(
    mut point_ptr: NonNull<[[u8; 32]; 2]>,
) -> bool {
    let x = Felt::from_bytes_le(&point_ptr.as_ref()[0]);

    // https://github.com/starkware-libs/cairo/blob/aaad921bba52e729dc24ece07fab2edf09ccfa15/crates/cairo-lang-sierra-to-casm/src/invocations/ec.rs#L63

    let x2 = x * x;
    let x3 = x2 * x;
    let alpha_x_plus_beta = x + BETA;
    let rhs = x3 + alpha_x_plus_beta;
    let y = rhs.sqrt().unwrap_or_else(|| Felt::from(3) * rhs);

    match AffinePoint::new(x, y) {
        Ok(point) => {
            point_ptr.as_mut()[1].copy_from_slice(&point.y().to_bytes_le());
            true
        }
        Err(_) => false,
    }
}

/// Compute `ec_point_try_new_nz(x)`.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_try_new_nz(
    mut point_ptr: NonNull<[[u8; 32]; 2]>,
) -> bool {
    let x = Felt::from_bytes_le(&point_ptr.as_ref()[0]);
    let y = Felt::from_bytes_le(&point_ptr.as_ref()[1]);

    match AffinePoint::new(x, y) {
        Ok(point) => {
            point_ptr.as_mut()[0].copy_from_slice(&point.x().to_bytes_le());
            point_ptr.as_mut()[1].copy_from_slice(&point.y().to_bytes_le());
            true
        }
        Err(_) => false,
    }
}

/// Compute `ec_state_init()` and store the state back.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_init(
    mut state_ptr: NonNull<[[u8; 32]; 4]>,
) {
    // https://github.com/starkware-libs/cairo/blob/aaad921bba52e729dc24ece07fab2edf09ccfa15/crates/cairo-lang-runner/src/casm_run/mod.rs#L1802
    let mut rng = rand::thread_rng();
    let (random_x, random_y) = loop {
        // Randominzing 31 bytes to make sure is in range.
        let x_bytes: [u8; 31] = rng.gen();
        let random_x = Felt::from_bytes_be_slice(&x_bytes);
        let random_y_squared = random_x * random_x * random_x + random_x + BETA;
        if let Some(random_y) = random_y_squared.sqrt() {
            break (random_x, random_y);
        }
    };

    let state = AffinePoint::new(random_x, random_y).unwrap();

    state_ptr.as_mut()[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr.as_mut()[1].copy_from_slice(&state.y().to_bytes_le());
    state_ptr.as_mut()[2].copy_from_slice(&state.x().to_bytes_le());
    state_ptr.as_mut()[3].copy_from_slice(&state.y().to_bytes_le());
}

/// Compute `ec_state_add(state, point)` and store the state back.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add(
    mut state_ptr: NonNull<[[u8; 32]; 4]>,
    point_ptr: NonNull<[[u8; 32]; 2]>,
) {
    let mut state = ProjectivePoint::from_affine(
        Felt::from_bytes_le(&state_ptr.as_ref()[0]),
        Felt::from_bytes_le(&state_ptr.as_ref()[1]),
    )
    .unwrap();
    let point = AffinePoint::new(
        Felt::from_bytes_le(&point_ptr.as_ref()[0]),
        Felt::from_bytes_le(&point_ptr.as_ref()[1]),
    )
    .unwrap();

    state += &point;
    let state = state.to_affine().unwrap();

    state_ptr.as_mut()[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr.as_mut()[1].copy_from_slice(&state.y().to_bytes_le());
}

/// Compute `ec_state_add_mul(state, scalar, point)` and store the state back.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add_mul(
    mut state_ptr: NonNull<[[u8; 32]; 4]>,
    scalar_ptr: NonNull<[u8; 32]>,
    point_ptr: NonNull<[[u8; 32]; 2]>,
) {
    let mut state = ProjectivePoint::from_affine(
        Felt::from_bytes_le(&state_ptr.as_ref()[0]),
        Felt::from_bytes_le(&state_ptr.as_ref()[1]),
    )
    .unwrap();
    let point = ProjectivePoint::from_affine(
        Felt::from_bytes_le(&point_ptr.as_ref()[0]),
        Felt::from_bytes_le(&point_ptr.as_ref()[1]),
    )
    .unwrap();
    let scalar = Felt::from_bytes_le(scalar_ptr.as_ref());

    state += &point.mul(scalar);
    let state = state.to_affine().unwrap();

    state_ptr.as_mut()[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr.as_mut()[1].copy_from_slice(&state.y().to_bytes_le());
}

/// Compute `ec_state_try_finalize_nz(state)` and store the result.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_try_finalize_nz(
    mut point_ptr: NonNull<[[u8; 32]; 2]>,
    state_ptr: NonNull<[[u8; 32]; 4]>,
) -> bool {
    let state = ProjectivePoint::from_affine(
        Felt::from_bytes_le(&state_ptr.as_ref()[0]),
        Felt::from_bytes_le(&state_ptr.as_ref()[1]),
    )
    .unwrap();
    let random = ProjectivePoint::from_affine(
        Felt::from_bytes_le(&state_ptr.as_ref()[2]),
        Felt::from_bytes_le(&state_ptr.as_ref()[3]),
    )
    .unwrap();

    if state.x() == random.x() && state.y() == random.y() {
        false
    } else {
        let point = &state - &random;
        let point = point.to_affine().unwrap();

        point_ptr.as_mut()[0].copy_from_slice(&point.x().to_bytes_le());
        point_ptr.as_mut()[1].copy_from_slice(&point.y().to_bytes_le());

        true
    }
}

#[cfg(feature = "with-trace-dump")]
pub mod trace_dump {
    use cairo_lang_sierra::{
        extensions::{
            core::{CoreLibfunc, CoreType, CoreTypeConcrete},
            starknet::StarkNetTypeConcrete,
        },
        ids::{ConcreteTypeId, VarId},
        program::StatementIdx,
        program_registry::ProgramRegistry,
    };
    use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
    use sierra_emu::{ProgramTrace, StateDump, Value};
    use starknet_crypto::Felt;
    use std::{
        alloc::Layout,
        collections::HashMap,
        mem::swap,
        ptr::NonNull,
        sync::{LazyLock, Mutex},
    };

    pub static TRACE_DUMP: LazyLock<Mutex<HashMap<u64, TraceDump>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    pub struct TraceDump {
        pub trace: ProgramTrace,
        state: OrderedHashMap<VarId, Value>,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,

        get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
    }

    impl TraceDump {
        pub fn new(
            registry: ProgramRegistry<CoreType, CoreLibfunc>,
            get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
        ) -> Self {
            Self {
                trace: ProgramTrace::default(),
                state: OrderedHashMap::default(),
                registry,

                get_layout,
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn get_trace_dump_ptr() -> *const Mutex<HashMap<u64, TraceDump>> {
        &*TRACE_DUMP as *const _
    }

    #[no_mangle]
    pub unsafe extern "C" fn cairo_native__trace_dump__state(
        trace_id: u64,
        var_id: u64,
        type_id: u64,
        value_ptr: NonNull<()>,
    ) {
        let mut trace_dump = TRACE_DUMP.lock().unwrap();
        let trace_dump = trace_dump.get_mut(&trace_id).unwrap();

        let type_id = ConcreteTypeId::new(type_id);
        let value = read_value_ptr(
            &trace_dump.registry,
            &type_id,
            value_ptr,
            trace_dump.get_layout,
        );

        trace_dump.state.insert(VarId::new(var_id), value);
    }

    #[no_mangle]
    pub unsafe extern "C" fn cairo_native__trace_dump__push(trace_id: u64, statement_idx: u64) {
        let mut trace_dump = TRACE_DUMP.lock().unwrap();
        let trace_dump = trace_dump.get_mut(&trace_id).unwrap();

        let mut items = OrderedHashMap::default();
        swap(&mut items, &mut trace_dump.state);

        trace_dump
            .trace
            .push(StateDump::new(StatementIdx(statement_idx as usize), items));
    }

    unsafe fn read_value_ptr(
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        type_id: &ConcreteTypeId,
        value_ptr: NonNull<()>,
        get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
    ) -> Value {
        let type_info = registry.get_type(type_id).unwrap();
        match type_info {
            CoreTypeConcrete::Felt252(_)
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::ContractAddress(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::ClassHash(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::StorageAddress(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::StorageBaseAddress(_)) => {
                Value::Felt(Felt::from_bytes_le(value_ptr.cast().as_ref()))
            }
            CoreTypeConcrete::Uint8(_) => Value::U8(value_ptr.cast().read()),
            CoreTypeConcrete::Uint16(_) => Value::U16(value_ptr.cast().read()),
            CoreTypeConcrete::Uint32(_) => Value::U32(value_ptr.cast().read()),
            CoreTypeConcrete::Uint64(_) => Value::U64(value_ptr.cast().read()),
            CoreTypeConcrete::Uint128(_) | CoreTypeConcrete::GasBuiltin(_) => {
                Value::U128(value_ptr.cast().read())
            }

            CoreTypeConcrete::EcPoint(_) => {
                let layout = Layout::new::<()>();
                let (x, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };

                Value::EcPoint { x, y }
            }
            CoreTypeConcrete::EcState(_) => {
                let layout = Layout::new::<()>();
                let (x0, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y0, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (x1, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y1, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };

                Value::EcState { x0, y0, x1, y1 }
            }

            CoreTypeConcrete::Uninitialized(info) => Value::Uninitialized {
                ty: info.ty.clone(),
            },
            CoreTypeConcrete::Box(info) => read_value_ptr(
                registry,
                &info.ty,
                value_ptr.cast::<NonNull<()>>().read(),
                get_layout,
            ),
            CoreTypeConcrete::Array(info) => {
                let layout = Layout::new::<()>();
                let (array_ptr, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<*mut ()>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<*mut ()>().read(), layout)
                };
                let (array_begin, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<u32>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<u32>().read(), layout)
                };
                let (array_end, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<u32>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<u32>().read(), layout)
                };

                let layout =
                    get_layout(registry.get_type(&info.ty).unwrap(), registry).pad_to_align();

                let mut data = Vec::with_capacity((array_end - array_begin) as usize);
                for index in array_begin..array_end {
                    let index = index as usize;

                    data.push(read_value_ptr(
                        registry,
                        &info.ty,
                        NonNull::new(array_ptr.byte_add(layout.size() * index)).unwrap(),
                        get_layout,
                    ));
                }

                Value::Array {
                    ty: info.ty.clone(),
                    data,
                }
            }

            CoreTypeConcrete::Struct(info) => {
                let mut layout = Layout::new::<()>();
                let mut members = Vec::with_capacity(info.members.len());
                for member_ty in &info.members {
                    let type_info = registry.get_type(member_ty).unwrap();
                    let member_layout = get_layout(type_info, registry);

                    let offset;
                    (layout, offset) = layout.extend(member_layout).unwrap();

                    let current_ptr = value_ptr.byte_add(offset);
                    members.push(read_value_ptr(registry, member_ty, current_ptr, get_layout));
                }

                Value::Struct(members)
            }
            CoreTypeConcrete::Enum(info) => {
                let num_variants = match info.variants.len() {
                    0 => unreachable!(),
                    1 => 0,
                    n => (n.next_power_of_two().next_multiple_of(8) >> 3) as _,
                };

                let layout = Layout::new::<()>();
                let (tag_value, layout) = match num_variants {
                    x if x <= 8 => {
                        let (layout, offset) = layout.extend(Layout::new::<u8>()).unwrap();
                        (
                            value_ptr.byte_add(offset).cast::<u8>().read() as usize,
                            layout,
                        )
                    }
                    x if x <= 16 => {
                        let (layout, offset) = layout.extend(Layout::new::<u16>()).unwrap();
                        (
                            value_ptr.byte_add(offset).cast::<u16>().read() as usize,
                            layout,
                        )
                    }
                    x if x <= 32 => {
                        let (layout, offset) = layout.extend(Layout::new::<u32>()).unwrap();
                        (
                            value_ptr.byte_add(offset).cast::<u32>().read() as usize,
                            layout,
                        )
                    }
                    x if x <= 64 => {
                        let (layout, offset) = layout.extend(Layout::new::<u64>()).unwrap();
                        (
                            value_ptr.byte_add(offset).cast::<u64>().read() as usize,
                            layout,
                        )
                    }
                    _ => todo!(),
                };

                let payload = {
                    let (_, offset) = layout
                        .extend(get_layout(
                            registry.get_type(&info.variants[tag_value]).unwrap(),
                            registry,
                        ))
                        .unwrap();

                    read_value_ptr(
                        registry,
                        &info.variants[tag_value],
                        value_ptr.byte_add(offset),
                        get_layout,
                    )
                };

                Value::Enum {
                    self_ty: type_id.clone(),
                    index: tag_value,
                    payload: Box::new(payload),
                }
            }

            CoreTypeConcrete::NonZero(info) | CoreTypeConcrete::Snapshot(info) => {
                read_value_ptr(registry, &info.ty, value_ptr, get_layout)
            }

            // Builtins and other unit types:
            CoreTypeConcrete::Bitwise(_)
            | CoreTypeConcrete::BuiltinCosts(_)
            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Poseidon(_)
            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::SegmentArena(_)
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_))
            | CoreTypeConcrete::Uint128MulGuarantee(_) => Value::Unit,

            // TODO:
            CoreTypeConcrete::Coupon(_) => todo!("CoreTypeConcrete::Coupon"),
            CoreTypeConcrete::Circuit(_) => todo!("CoreTypeConcrete::Circuit"),
            CoreTypeConcrete::Const(_) => todo!("CoreTypeConcrete::Const"),
            CoreTypeConcrete::Sint8(_) => todo!("CoreTypeConcrete::Sint8"),
            CoreTypeConcrete::Sint16(_) => todo!("CoreTypeConcrete::Sint16"),
            CoreTypeConcrete::Sint32(_) => todo!("CoreTypeConcrete::Sint32"),
            CoreTypeConcrete::Sint64(_) => todo!("CoreTypeConcrete::Sint64"),
            CoreTypeConcrete::Sint128(_) => todo!("CoreTypeConcrete::Sint128"),
            CoreTypeConcrete::Nullable(_) => todo!("CoreTypeConcrete::Nullable"),
            CoreTypeConcrete::RangeCheck96(_) => todo!("CoreTypeConcrete::RangeCheck96"),
            CoreTypeConcrete::Felt252Dict(_) => todo!("CoreTypeConcrete::Felt252Dict"),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!("CoreTypeConcrete::Felt252DictEntry"),
            CoreTypeConcrete::SquashedFelt252Dict(_) => {
                todo!("CoreTypeConcrete::SquashedFelt252Dict")
            }
            CoreTypeConcrete::Span(_) => todo!("CoreTypeConcrete::Span"),
            CoreTypeConcrete::StarkNet(selector) => match selector {
                StarkNetTypeConcrete::Secp256Point(_) => {
                    todo!("StarkNetTypeConcrete::Secp256Point")
                }
                StarkNetTypeConcrete::Sha256StateHandle(_) => {
                    todo!("StarkNetTypeConcrete::Sha256StateHandle")
                }
                _ => unreachable!(),
            },
            CoreTypeConcrete::Bytes31(_) => todo!("CoreTypeConcrete::Bytes31"),
            CoreTypeConcrete::BoundedInt(_) => todo!("CoreTypeConcrete::BoundedInt"),
        }
    }
}
