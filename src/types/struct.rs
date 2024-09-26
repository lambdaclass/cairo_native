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
    metadata::{snapshot_clones::SnapshotClonesMeta, MetadataStorage},
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
    dialect::llvm,
    ir::{Block, Location, Module, Type, Value},
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
    let fields: Vec<_> = info
        .members
        .iter()
        .map(|field| registry.build_type(context, module, registry, metadata, field))
        .collect::<Result<_>>()?;
    let struct_ty = llvm::r#type::r#struct(context, &fields, false);

    if let Some(snapshot_clones_meta) = metadata.get_mut::<SnapshotClonesMeta>() {
        if info
            .members
            .iter()
            .any(|ty| snapshot_clones_meta.wrap_invoke(ty).is_some())
        {
            snapshot_clones_meta.register(
                info.self_ty.clone(),
                snapshot_take,
                StructConcreteType {
                    info: info.info.clone(),
                    members: info.members.clone(),
                },
            );
        }
    }

    Ok(struct_ty)
}

#[allow(clippy::too_many_arguments)]
fn snapshot_take<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    mut entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: WithSelf<StructConcreteType>,
    src_value: Value<'ctx, 'this>,
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
    let self_ty = registry.build_type(context, helper, registry, metadata, info.self_ty)?;
    let mut value = entry.append_op_result(llvm::undef(self_ty, location))?;

    // The following unwrap is unreachable because we've already check that at least one of the
    // struct's members have a custom clone implementation before registering this function.
    let snapshot_clones_meta = metadata.get::<SnapshotClonesMeta>().unwrap();
    for (member_idx, member_id) in info.members.iter().enumerate() {
        let member_ty = registry.build_type(context, helper, registry, metadata, member_id)?;
        let member_val =
            entry.extract_value(context, location, src_value, member_ty, member_idx)?;

        let cloned_member_val;
        (entry, cloned_member_val) = match snapshot_clones_meta.wrap_invoke(member_id) {
            Some(clone_fn) => clone_fn(
                context, registry, entry, location, helper, metadata, member_val,
            )?,
            None => (entry, member_val),
        };

        value = entry.insert_value(context, location, value, cloned_member_val, member_idx)?;
    }

    Ok((entry, value))
}

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
    let value = entry.argument(0)?.into();

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
