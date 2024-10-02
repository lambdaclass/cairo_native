//! # Struct type
//!
//! A struct is just a fixed collection of values that may have different types, which are known at
//! compile-time. Its fields are properly aligned and respect the declaration's field ordering.
//!
//! For example, the following struct would have a layout as described in the table below:
//!
//! ```cairo
//! struct MyStruct {
//!     U8: u8,
//!     U16: u16,
//!     U32: u32,
//!     U64: u64,
//!     Felt: Felt,
//! }
//! ```
//!
//! | Index | Type   | ABI (in Rust types) | Alignment | Size |
//! | ----- | ------ | ------------------- | --------- | ---- |
//! |   0   | `i8`   | `u8`                |         1 |    1 |
//! |  N/A  | N/A    | `[u8; 1]`           |         1 |    1 |
//! |   1   | `i16`  | `u16`               |         2 |    2 |
//! |  N/A  | N/A    | `[u8; 2]`           |         1 |    2 |
//! |   2   | `i32`  | `u32`               |         4 |    4 |
//! |  N/A  | N/A    | `[u8; 4]`           |         1 |    4 |
//! |   3   | `i64`  | `u64`               |         8 |    8 |
//! |   4   | `i252` | `[u64; 4]`          |         8 |    8 |
//!
//! As inferred in the table above, the struct will have 8-byte alignment and a size of 30 bytes.
//! Since this way of generating structs is equivalent to the one used in C and C++, the same
//! effects apply. For example, if we invert the order of the fields the ABI will change but we
//! won't waste a single byte in padding; unless we're creating an array, in which case we'd waste
//! only a single byte per element.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    libfuncs::LibfuncHelper,
    metadata::{dup_overrides::DupOverridesMeta, MetadataStorage},
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        structure::StructConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{func, llvm},
    ir::{Block, Location, Module, Region, Type, Value},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<StructConcreteType>,
) -> Result<Type<'ctx>> {
    DupOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // The following unwrap is unreachable because `register_with` will always insert it
            // before calling this closure.
            let mut needs_clone_override = false;
            for member in &info.members {
                registry.build_type(context, module, registry, metadata, member)?;
                if metadata
                    .get::<DupOverridesMeta>()
                    .unwrap()
                    .is_overriden(member)
                {
                    needs_clone_override = true;
                    break;
                }
            }

            needs_clone_override
                .then(|| build_dup(context, module, registry, metadata, &info))
                .transpose()
        },
    )?;

    let members = info
        .members
        .iter()
        .map(|member| registry.build_type(context, module, registry, metadata, member))
        .collect::<Result<Vec<_>>>()?;
    Ok(llvm::r#type::r#struct(context, &members, false))
}

fn build_dup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<StructConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);

    let self_ty = registry.build_type(context, module, registry, metadata, info.self_ty())?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(self_ty, location)]));

    let mut src_value = entry.argument(0)?.into();
    let mut dst_value = entry.append_op_result(llvm::undef(self_ty, location))?;

    for (idx, member_id) in info.members.iter().enumerate() {
        let member_ty = registry.build_type(context, module, registry, metadata, member_id)?;
        let member_val = entry.extract_value(context, location, src_value, member_ty, idx)?;

        // The following unwrap is unreachable because the registration logic will always insert it.
        let values = metadata
            .get::<DupOverridesMeta>()
            .unwrap()
            .invoke_override(context, &entry, location, member_id, member_val)?;

        src_value = entry.insert_value(context, location, src_value, values.0, idx)?;
        dst_value = entry.insert_value(context, location, dst_value, values.1, idx)?;
    }

    entry.append_operation(func::r#return(&[src_value, dst_value], location));
    Ok(region)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_drop<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: WithSelf<StructConcreteType>,
    value: Value<'ctx, 'this>,
) -> Result<()> {
    // Since we don't currently have a way to check if a type should implement drop or not we just
    // call `build_drop` for every member. The canonicalization pass should remove unnecessary
    // `extractvalue` operations.
    for (member_idx, member_id) in info.members.iter().enumerate() {
        let member_ty = registry.build_type(context, helper, registry, metadata, member_id)?;
        let member_val = entry.extract_value(context, location, value, member_ty, member_idx)?;

        registry.get_type(member_id)?.build_drop(
            context, registry, entry, location, helper, metadata, member_id, member_val,
        )?;
    }

    Ok(())
}
