//! # Struct-related libfuncs

use std::alloc::Layout;

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    native_panic,
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureOnlyConcreteLibfunc,
        structure::{ConcreteStructBoxedDeconstructLibfunc, StructConcreteLibfunc},
        ConcreteLibfunc,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use itertools::Itertools;
use melior::{
    dialect::{
        llvm::{self, r#type::pointer, LoadStoreOptions},
        ods,
    },
    helpers::{ArithBlockExt, BuiltinBlockExt, LlvmBlockExt},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, BlockLike, Location, Value},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &StructConcreteLibfunc,
) -> Result<()> {
    match selector {
        StructConcreteLibfunc::Construct(info) => {
            build_construct(context, registry, entry, location, helper, metadata, info)
        }
        StructConcreteLibfunc::Deconstruct(info)
        | StructConcreteLibfunc::SnapshotDeconstruct(info) => {
            build_deconstruct(context, registry, entry, location, helper, metadata, info)
        }
        StructConcreteLibfunc::BoxedDeconstruct(info) => {
            build_boxed_deconstruct(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `struct_construct` libfunc.
pub fn build_construct<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let mut fields = Vec::new();

    for (i, _) in info.param_signatures().iter().enumerate() {
        fields.push(entry.argument(i)?.into());
    }

    let value = build_struct_value(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
        &fields,
    )?;

    helper.br(entry, 0, &[value], location)
}

/// Generate MLIR operations for the `struct_construct` libfunc.
#[allow(clippy::too_many_arguments)]
pub fn build_struct_value<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    struct_type: &ConcreteTypeId,
    fields: &[Value<'ctx, 'this>],
) -> Result<Value<'ctx, 'this>> {
    let struct_ty = registry.build_type(context, helper, metadata, struct_type)?;

    let struct_type = registry.get_type(struct_type)?;

    // LLVM fails when inserting zero-sized types into a struct.
    // See: https://github.com/llvm/llvm-project/issues/107198.
    // We will manually skip ZST fields, as inserting a ZST is a noop, and there
    // is no point on building that operation.
    let zst_fields = match struct_type {
        CoreTypeConcrete::Struct(info) => {
            // If the type is a struct, we check for each member if its a ZST.
            info.members
                .iter()
                .map(|member| registry.get_type(member)?.is_zst(registry))
                .try_collect()?
        }
        _ => {
            // There are many Sierra types represented as an LLVM struct, but
            // are not of Sierra struct type. In these cases we assume that
            // their members are not ZST.
            vec![false; fields.len()]
        }
    };

    let mut accumulator = entry.append_op_result(llvm::undef(struct_ty, location))?;
    for (idx, field) in fields.iter().enumerate() {
        if zst_fields[idx] {
            continue;
        }

        accumulator = entry.insert_value(context, location, accumulator, *field, idx)?
    }

    Ok(accumulator)
}

/// Generate MLIR operations for the `struct_deconstruct` libfunc.
pub fn build_deconstruct<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let container = entry.arg(0)?;

    let mut fields = Vec::<Value>::with_capacity(info.branch_signatures()[0].vars.len());
    for (i, var_info) in info.branch_signatures()[0].vars.iter().enumerate() {
        let type_info = registry.get_type(&var_info.ty)?;
        let field_ty = type_info.build(context, helper, registry, metadata, &var_info.ty)?;

        let value = entry.extract_value(context, location, container, field_ty, i)?;

        fields.push(value);
    }

    helper.br(entry, 0, &fields, location)
}

fn unbox_container<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConcreteStructBoxedDeconstructLibfunc,
) -> Result<Value<'ctx, 'this>> {
    let boxed_container_ty_id = &info.param_signatures()[0].ty;

    let (container_ty, container_layout) =
        if let CoreTypeConcrete::Box(box_info) = registry.get_type(boxed_container_ty_id)? {
            registry.build_type_with_layout(context, helper, metadata, &box_info.ty)?
        } else {
            native_panic!("Should receive a boxed Struct");
        };

    let value = entry
        .append_operation(llvm::load(
            context,
            entry.arg(0)?,
            container_ty,
            location,
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                IntegerType::new(context, 64).into(),
                container_layout.align() as i64,
            ))),
        ))
        .result(0)?
        .into();

    Ok(value)
}

fn box_member<'ctx, 'this>(
    context: &'ctx Context,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    member: Value<'ctx, 'this>,
    member_layout: Layout,
) -> Result<Value<'ctx, 'this>> {
    let len = entry.const_int(context, location, member_layout.pad_to_align().size(), 64)?;
    let ptr = entry
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .result(0)?
        .into();
    let ptr = entry
        .append_operation(ReallocBindingsMeta::realloc(context, ptr, len, location)?)
        .result(0)?
        .into();

    entry.store(context, location, ptr, member)?;

    Ok(ptr)
}

pub fn build_boxed_deconstruct<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConcreteStructBoxedDeconstructLibfunc,
) -> Result<()> {
    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    // Unbox the container
    let container = unbox_container(context, registry, entry, location, helper, metadata, info)?;

    let mut fields = Vec::<Value>::with_capacity(info.members.len());
    for (i, member_type_id) in info.members.iter().enumerate() {
        let type_info = registry.get_type(member_type_id)?;
        let field_ty = type_info.build(context, helper, registry, metadata, member_type_id)?;

        let member = entry.extract_value(context, location, container, field_ty, i)?;
        let (_, member_layout) =
            registry.build_type_with_layout(context, helper, metadata, member_type_id)?;
        // Box the member
        let member = box_member(context, entry, location, member, member_layout)?;

        fields.push(member);
    }

    // Free the boxed container
    entry.append_operation(ReallocBindingsMeta::free(context, entry.arg(0)?, location)?);

    helper.br(entry, 0, &fields, location)
}

#[cfg(test)]
mod test {
    use crate::{jit_struct, load_cairo, utils::testing::run_program_assert_output, Value};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;

    lazy_static! {
        static ref BOXED_DECONSTRUCT_PROGRAM: (String, Program) = load_cairo! {
            mod decons_3_fields {
                extern fn struct_boxed_deconstruct<T>(value: Box<T>) -> (Box<felt252>, Box<u8>, Box<u128>) nopanic;
            }

            mod decons_1_field {
                extern fn struct_boxed_deconstruct<T>(value: Box<T>) -> (Box<u8>,) nopanic;
            }

            mod decons_empty_struct {
                extern fn struct_boxed_deconstruct<T>(value: Box<T>) -> () nopanic;
            }

            mod decons_struct_snapshot {
                extern fn struct_boxed_deconstruct<T>(value: Box<T>) -> (Box<@felt252>, Box<@u8>, Box<@u128>) nopanic;
            }

            struct ThreeFields {
                x: felt252,
                y: u8,
                z: u128,
            }

            struct OneField {
                x: u8,
            }

            struct EmptyStruct { }

            fn deconstruct_struct_3_fields() -> (Box<felt252>, Box<u8>, Box<u128>) {
                decons_3_fields::struct_boxed_deconstruct(BoxTrait::new(ThreeFields {x: 2, y: 2, z: 2}))
            }

            fn deconstruct_struct_1_field() -> (Box<u8>,) {
                decons_1_field::struct_boxed_deconstruct(BoxTrait::new(OneField {x: 2}))
            }

            fn deconstruct_empty_struct() -> () {
                decons_empty_struct::struct_boxed_deconstruct(BoxTrait::new(EmptyStruct { }))
            }

            fn deconstruct_struct_snapshot() -> (Box<@felt252>, Box<@u8>, Box<@u128>) {
                decons_struct_snapshot::struct_boxed_deconstruct(BoxTrait::new(ThreeFields {x: 2, y: 2, z: 2}))
            }
        };
    }

    #[test]
    fn boxed_deconstruct_3_fields() {
        run_program_assert_output(
            &BOXED_DECONSTRUCT_PROGRAM,
            "deconstruct_struct_3_fields",
            &[],
            jit_struct!(Value::Felt252(2.into()), Value::Uint8(2), Value::Uint128(2)),
        );
    }

    #[test]
    fn boxed_deconstruct_1_field() {
        run_program_assert_output(
            &BOXED_DECONSTRUCT_PROGRAM,
            "deconstruct_struct_1_field",
            &[],
            jit_struct!(Value::Uint8(2)),
        );
    }

    #[test]
    fn boxed_deconstruct_empty_struct() {
        run_program_assert_output(
            &BOXED_DECONSTRUCT_PROGRAM,
            "deconstruct_empty_struct",
            &[],
            jit_struct!(),
        );
    }

    #[test]
    fn boxed_deconstruct_struct_snapshot() {
        run_program_assert_output(
            &BOXED_DECONSTRUCT_PROGRAM,
            "deconstruct_struct_snapshot",
            &[],
            jit_struct!(Value::Felt252(2.into()), Value::Uint8(2), Value::Uint128(2)),
        );
    }
}
