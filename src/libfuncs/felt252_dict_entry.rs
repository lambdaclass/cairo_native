//! # `Felt` dictionary entry libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{
        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
        MetadataStorage,
    },
    types::TypeBuilder,
    utils::{get_integer_layout, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        felt252_dict::Felt252DictEntryConcreteLibfunc, lib_func::SignatureAndTypeConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith, cf,
        llvm::{self, r#type::opaque_pointer, LoadStoreOptions},
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location, Value, ValueLike,
    },
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252DictEntryConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Felt252DictEntryConcreteLibfunc::Get(info) => {
            build_get(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictEntryConcreteLibfunc::Finalize(info) => {
            build_finalize(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_get<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let (key_ty, key_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;

    let entry_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let (value_ty, value_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let dict_ptr = entry.argument(0)?.into();
    let key_value = entry.argument(1)?.into();

    let const_1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let op = helper.init_block().append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    key_layout.align().try_into()?,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[const_1])
            .add_results(&[llvm::r#type::pointer(key_ty, 0)])
            .build()?,
    );

    let key_ptr = op.result(0)?.into();

    entry.append_operation(llvm::store(
        context,
        key_value,
        key_ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            key_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let op = runtime_bindings.dict_get(context, helper, entry, dict_ptr, key_ptr, location)?;
    let result_ptr: Value = op.result(0)?.into();

    let op = entry.append_operation(
        OperationBuilder::new("llvm.mlir.null", location)
            .add_results(&[result_ptr.r#type()])
            .build()?,
    );

    let null_ptr = op.result(0)?.into();

    // need llvm instead of arith to compare pointers
    let op = entry.append_operation(
        OperationBuilder::new("llvm.icmp", location)
            .add_operands(&[result_ptr, null_ptr])
            .add_attributes(&[(
                Identifier::new(context, "predicate"),
                IntegerAttribute::new(0, IntegerType::new(context, 64).into()).into(),
            )])
            .add_results(&[IntegerType::new(context, 1).into()])
            .build()?,
    );

    let is_null_ptr = op.result(0)?.into();

    let block_is_null = helper.append_block(Block::new(&[]));
    let block_is_found = helper.append_block(Block::new(&[]));
    let block_final = helper.append_block(Block::new(&[
        (opaque_pointer(context), location),
        (value_ty, location),
    ]));

    entry.append_operation(cf::cond_br(
        context,
        is_null_ptr,
        block_is_null,
        block_is_found,
        &[],
        &[],
        location,
    ));

    // null block
    {
        let op = block_is_null.append_operation(arith::constant(
            context,
            IntegerAttribute::new(
                value_layout.size() as i64,
                IntegerType::new(context, 64).into(),
            )
            .into(),
            location,
        ));
        let alloc_size = op.result(0)?.into();

        let op = block_is_null.append_operation(ReallocBindingsMeta::realloc(
            context, result_ptr, alloc_size, location,
        ));
        let value_ptr = op.result(0)?.into();

        let op = block_is_null.append_operation(llvm::undef(value_ty, location));
        let undef_value = op.result(0)?.into();

        block_is_null.append_operation(cf::br(block_final, &[value_ptr, undef_value], location));
    }

    // found block
    {
        let op = block_is_found.append_operation(llvm::load(
            context,
            result_ptr,
            value_ty,
            location,
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                value_layout.align() as i64,
                IntegerType::new(context, 64).into(),
            ))),
        ));
        let loaded_value = op.result(0)?.into();
        block_is_found.append_operation(cf::br(block_final, &[result_ptr, loaded_value], location));
    }

    // construct the struct

    let op = block_final.append_operation(llvm::undef(entry_ty, location));
    let entry_value = op.result(0)?.into();

    let value_ptr = block_final.argument(0)?.into();
    let value = block_final.argument(1)?.into();

    let op = block_final.append_operation(llvm::insert_value(
        context,
        entry_value,
        DenseI64ArrayAttribute::new(context, &[0]),
        key_value,
        location,
    ));
    let entry_value = op.result(0)?.into();
    let op = block_final.append_operation(llvm::insert_value(
        context,
        entry_value,
        DenseI64ArrayAttribute::new(context, &[1]),
        value_ptr,
        location,
    ));
    let entry_value = op.result(0)?.into();
    let op = block_final.append_operation(llvm::insert_value(
        context,
        entry_value,
        DenseI64ArrayAttribute::new(context, &[2]),
        dict_ptr,
        location,
    ));
    let entry_value = op.result(0)?.into();

    block_final.append_operation(helper.br(0, &[entry_value, value], location));

    Ok(())
}

pub fn build_finalize<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value_type = registry.get_type(&info.param_signatures()[1].ty)?;
    let value_layout = value_type.layout(registry)?;

    let key_ty = IntegerType::new(context, 252).into();
    let key_layout = get_integer_layout(252);

    let entry_value = entry.argument(0)?.into();
    let new_value = entry.argument(1)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        entry_value,
        DenseI64ArrayAttribute::new(context, &[0]),
        key_ty,
        location,
    ));
    let key_value = op.result(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        entry_value,
        DenseI64ArrayAttribute::new(context, &[1]),
        opaque_pointer(context),
        location,
    ));
    let value_ptr = op.result(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        entry_value,
        DenseI64ArrayAttribute::new(context, &[2]),
        opaque_pointer(context),
        location,
    ));
    let dict_ptr = op.result(0)?.into();

    entry.append_operation(llvm::store(
        context,
        new_value,
        value_ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            value_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let const_1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let op = helper.init_block().append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    key_layout.align().try_into()?,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[const_1])
            .add_results(&[llvm::r#type::pointer(key_ty, 0)])
            .build()?,
    );

    let key_ptr = op.result(0)?.into();

    entry.append_operation(llvm::store(
        context,
        key_value,
        key_ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            key_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    // call insert

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    runtime_bindings.dict_insert(
        context, helper, entry, dict_ptr, key_ptr, value_ptr, location,
    )?;

    entry.append_operation(helper.br(0, &[dict_ptr], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_dict, load_cairo, run_program_assert_output};

    #[test]
    fn run_dict_insert() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u32 {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict.get(2)
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[1u32.into()]);
    }

    #[test]
    fn run_dict_insert_big() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u64 {
                let mut dict: Felt252Dict<u64> = Default::default();
                dict.insert(200000000, 4_u64);
                dict.get(200000000)
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[4u64.into()]);
    }

    #[test]
    fn run_dict_insert_ret_dict() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> Felt252Dict<u32> {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            &[jit_dict!(
                2 => 1u32
            )],
        );
    }

    #[test]
    fn run_dict_insert_multiple() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u32 {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict.insert(3, 1_u32);
                dict.insert(4, 1_u32);
                dict.insert(5, 1_u32);
                dict.insert(6, 1_u32);
                dict.insert(7, 1_u32);
                dict.insert(8, 1_u32);
                dict.insert(9, 1_u32);
                dict.insert(10, 1_u32);
                dict.insert(11, 1_u32);
                dict.insert(12, 1_u32);
                dict.insert(13, 1_u32);
                dict.insert(14, 1_u32);
                dict.insert(15, 1_u32);
                dict.insert(16, 1_u32);
                dict.insert(17, 1_u32);
                dict.insert(18, 1345432_u32);
                dict.get(18)
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[1345432_u32.into()]);
    }
}
