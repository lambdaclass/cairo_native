use cairo_lang_sierra::{
    extensions::{
        blake::BlakeConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{r#type::IntegerType, Block, BlockLike, Location, Value},
    dialect::{arith, llvm},
    ir::{r#type::IntegerType, Block, BlockLike, Location, Value},
    Context,
};

use crate::{
    error::Result,
    metadata::MetadataStorage,
    utils::{BlockExt, GepIndex},
};
use crate::{
    error::Result,
    metadata::MetadataStorage,
    utils::{BlockExt, GepIndex},
};

use super::LibfuncHelper;

// Used when initializing the v vector
pub const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const SIGMA: [[u32; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

// Used when initializing the v vector
pub const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const SIGMA: [[u32; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &BlakeConcreteLibfunc,
) -> Result<()> {
    match selector {
        BlakeConcreteLibfunc::Blake2sCompress(info) => build_blake_operation(
            context, registry, entry, location, helper, metadata, info, false,
        )?,
        BlakeConcreteLibfunc::Blake2sFinalize(info) => build_blake_operation(
            context, registry, entry, location, helper, metadata, info, true,
        )?,
        BlakeConcreteLibfunc::Blake2sCompress(info) => build_blake_operation(
            context, registry, entry, location, helper, metadata, info, false,
        )?,
        BlakeConcreteLibfunc::Blake2sFinalize(info) => build_blake_operation(
            context, registry, entry, location, helper, metadata, info, true,
        )?,
    }

    Ok(())
}

fn build_blake_operation<'ctx, 'this>(
fn build_blake_operation<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
    finalize: bool,
    finalize: bool,
) -> Result<()> {
    let curr_state_ptr = entry.arg(0)?;
    let bytes_count = entry.arg(1)?;
    let message = entry.arg(2)?;
    let k0 = entry.const_int(context, location, 0, 32)?;
    let kffffffff = entry.const_int(context, location, 0xffffffff as u32, 32)?;
    let u32_ty = IntegerType::new(context, 32).into();

    // IV values to MLIR values
    let iv_values = IV
        .into_iter()
        .map(|v| entry.const_int(context, location, v, 32))
        .collect::<Result<Vec<_>>>()?;

    // local state vector to work with during the compression
    let mut vector_v = Vec::with_capacity(16);

    // fill v according to the algorithm
    for i in 0..8 {
        let value = entry.gep(
            context,
            location,
            curr_state_ptr,
            &[GepIndex::Const(i)],
            u32_ty,
        )?;

        vector_v.push(value);
    }
    for i in 0..4 {
        vector_v.push(iv_values[i]);
    }
    vector_v.push(entry.append_op_result(arith::xori(iv_values[4], bytes_count, location))?);
    vector_v.push(entry.append_op_result(arith::xori(iv_values[5], k0, location))?);
    vector_v.push(entry.append_op_result(arith::xori(
        iv_values[6],
        if finalize { kffffffff } else { k0 },
        location,
    ))?);
    vector_v.push(entry.append_op_result(arith::xori(iv_values[7], k0, location))?);

    // Blake rounding
    for sigma_array in SIGMA {
        vector_v = blake_rounding(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            info,
            vector_v,
            message,
            sigma_array,
        )?;
    }

    let new_state =
        entry.append_op_result(llvm::undef(llvm::r#type::array(u32_ty, 8), location))?;

    // Create the new blake state
    // each new element in the state created by the formula
    // curr_state[i] ^ v[i] ^ v[8 + i]
    for i in 0..8 as usize {
        let state_value = entry.gep(
            context,
            location,
            curr_state_ptr,
            &[GepIndex::Const(i as i32)],
            u32_ty,
        )?;
        let xor_result = entry.append_op_result(arith::xori(state_value, vector_v[i], location))?;
        let xor_result =
            entry.append_op_result(arith::xori(xor_result, vector_v[8 + i], location))?;

        entry.insert_value(context, location, new_state, xor_result, i)?;
    }

    entry.append_operation(helper.br(0, &[new_state], location));

    Ok(())
}

fn blake_rounding<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
    vector_v: Vec<Value<'ctx, 'this>>,
    msg: Value<'ctx, 'this>,
    sigma_array: [u32; 16],
) -> Result<Vec<Value<'ctx, 'this>>> {
    let u32_ty = IntegerType::new(context, 32).into();

    (vector_v[0], vector_v[4], vector_v[8], vector_v[12]) = blak(
        vector_v[0],
        vector_v[4],
        vector_v[8],
        vector_v[12],
        entry.gep(context, location, msg, &[GepIndex::Const(sigma_array[0] as i32)], u32_ty),
        entry.gep(context, location, msg, &[GepIndex::Const(sigma_array[1] as i32)], u32_ty,
    );
    (vector_v[1], vector_v[5], vector_v[9], vector_v[13]) = mix(
        vector_v[1],
        vector_v[5],
        vector_v[9],
        vector_v[13],
        msg[sigma_array[2]],
        msg[sigma_array[3]],
    );
    (vector_v[2], vector_v[6], vector_v[10], vector_v[14]) = mix(
        vector_v[2],
        vector_v[6],
        vector_v[10],
        vector_v[14],
        msg[sigma_array[4]],
        msg[sigma_array[5]],
    );
    (vector_v[3], vector_v[7], vector_v[11], vector_v[15]) = mix(
        vector_v[3],
        vector_v[7],
        vector_v[11],
        vector_v[15],
        msg[sigma_array[6]],
        msg[sigma_array[7]],
    );
    (vector_v[0], vector_v[5], vector_v[10], vector_v[15]) = mix(
        vector_v[0],
        vector_v[5],
        vector_v[10],
        vector_v[15],
        msg[sigma_array[8]],
        msg[sigma_array[9]],
    );
    (vector_v[1], vector_v[6], vector_v[11], vector_v[12]) = mix(
        vector_v[1],
        vector_v[6],
        vector_v[11],
        vector_v[12],
        msg[sigma_array[10]],
        msg[sigma_array[11]],
    );
    (vector_v[2], vector_v[7], vector_v[8], vector_v[13]) = mix(
        vector_v[2],
        vector_v[7],
        vector_v[8],
        vector_v[13],
        msg[sigma_array[12]],
        msg[sigma_array[13]],
    );
    (vector_v[3], vector_v[4], vector_v[9], vector_v[14]) = mix(
        vector_v[3],
        vector_v[4],
        vector_v[9],
        vector_v[14],
        msg[sigma_array[14]],
        msg[sigma_array[15]],
    );
    Ok(vector_v)
}

fn blake_mix<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
    vector_v: Vec<Value<'ctx, 'this>>,
    msg: Value<'ctx, 'this>,
    sigma_array: Vec<Value<'ctx, 'this>>,
) -> Result<Vec<Value<'ctx, 'this>>> {
    todo!()
}

#[cfg(test)]
mod tests {
    use crate::{
        utils::test::{load_cairo, run_program},
        Value,
    };

    #[test]
    fn test_blake() {
        let program = load_cairo!(
            use core::blake::{blake2s_compress, blake2s_finalize};

            fn run_test() -> [u32; 8] nopanic {
                let state = BoxTrait::new([0_u32; 8]);
                let msg = BoxTrait::new([0_u32; 16]);
                let byte_count = 64_u32;

                let _res = blake2s_compress(state, byte_count, msg).unbox();

                blake2s_finalize(state, byte_count, msg).unbox()
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            Value::Array(vec![
                Value::Uint32(128291589),
                Value::Uint32(1454945417),
                Value::Uint32(3191583614),
                Value::Uint32(1491889056),
                Value::Uint32(794023379),
                Value::Uint32(651000200),
                Value::Uint32(3725903680),
                Value::Uint32(1044330286),
            ])
        );
    }
    vector_v: Vec<Value<'ctx, 'this>>,
    msg: Value<'ctx, 'this>,
    sigma_array: Vec<Value<'ctx, 'this>>,
) -> Result<Vec<Value<'ctx, 'this>>> {
    todo!()
}

#[cfg(test)]
mod tests {
    use crate::{
        utils::test::{load_cairo, run_program},
        Value,
    };

    #[test]
    fn test_blake() {
        let program = load_cairo!(
            use core::blake::{blake2s_compress, blake2s_finalize};

            fn run_test() -> [u32; 8] nopanic {
                let state = BoxTrait::new([0_u32; 8]);
                let msg = BoxTrait::new([0_u32; 16]);
                let byte_count = 64_u32;

                let _res = blake2s_compress(state, byte_count, msg).unbox();

                blake2s_finalize(state, byte_count, msg).unbox()
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            Value::Array(vec![
                Value::Uint32(128291589),
                Value::Uint32(1454945417),
                Value::Uint32(3191583614),
                Value::Uint32(1491889056),
                Value::Uint32(794023379),
                Value::Uint32(651000200),
                Value::Uint32(3725903680),
                Value::Uint32(1044330286),
            ])
        );
    }
}
