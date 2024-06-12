////! # `Felt` dictionary entry libfuncs
//! # `Felt` dictionary entry libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::Result,
    error::Result,
//    metadata::{
    metadata::{
//        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
//        MetadataStorage,
        MetadataStorage,
//    },
    },
//    types::TypeBuilder,
    types::TypeBuilder,
//    utils::{get_integer_layout, ProgramRegistryExt},
    utils::{get_integer_layout, ProgramRegistryExt},
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        felt252_dict::Felt252DictEntryConcreteLibfunc,
        felt252_dict::Felt252DictEntryConcreteLibfunc,
//        lib_func::SignatureAndTypeConcreteLibfunc,
        lib_func::SignatureAndTypeConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{cf, llvm},
    dialect::{cf, llvm},
//    ir::{
    ir::{
//        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
//        Identifier, Location, Value, ValueLike,
        Identifier, Location, Value, ValueLike,
//    },
    },
//    Context,
    Context,
//};
};
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &Felt252DictEntryConcreteLibfunc,
    selector: &Felt252DictEntryConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        Felt252DictEntryConcreteLibfunc::Get(info) => {
        Felt252DictEntryConcreteLibfunc::Get(info) => {
//            build_get(context, registry, entry, location, helper, metadata, info)
            build_get(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Felt252DictEntryConcreteLibfunc::Finalize(info) => {
        Felt252DictEntryConcreteLibfunc::Finalize(info) => {
//            build_finalize(context, registry, entry, location, helper, metadata, info)
            build_finalize(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

//pub fn build_get<'ctx, 'this>(
pub fn build_get<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let (key_ty, key_layout) = registry.build_type_with_layout(
    let (key_ty, key_layout) = registry.build_type_with_layout(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[1].ty,
        &info.param_signatures()[1].ty,
//    )?;
    )?;
//

//    let entry_ty = registry.build_type(
    let entry_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//

//    let (value_ty, value_layout) = registry.build_type_with_layout(
    let (value_ty, value_layout) = registry.build_type_with_layout(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[1].ty,
        &info.branch_signatures()[0].vars[1].ty,
//    )?;
    )?;
//

//    let dict_ptr = entry.argument(0)?.into();
    let dict_ptr = entry.argument(0)?.into();
//    let key_value = entry.argument(1)?.into();
    let key_value = entry.argument(1)?.into();
//

//    let key_ptr =
    let key_ptr =
//        helper
        helper
//            .init_block()
            .init_block()
//            .alloca1(context, location, key_ty, Some(key_layout.align()))?;
            .alloca1(context, location, key_ty, Some(key_layout.align()))?;
//

//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        key_ptr,
        key_ptr,
//        key_value,
        key_value,
//        Some(key_layout.align()),
        Some(key_layout.align()),
//    );
    );
//

//    let runtime_bindings = metadata
    let runtime_bindings = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .expect("Runtime library not available.");
        .expect("Runtime library not available.");
//

//    let op = runtime_bindings.dict_get(context, helper, entry, dict_ptr, key_ptr, location)?;
    let op = runtime_bindings.dict_get(context, helper, entry, dict_ptr, key_ptr, location)?;
//    let result_ptr: Value = op.result(0)?.into();
    let result_ptr: Value = op.result(0)?.into();
//

//    let null_ptr = entry.append_op_result(
    let null_ptr = entry.append_op_result(
//        OperationBuilder::new("llvm.mlir.zero", location)
        OperationBuilder::new("llvm.mlir.zero", location)
//            .add_results(&[result_ptr.r#type()])
            .add_results(&[result_ptr.r#type()])
//            .build()?,
            .build()?,
//    )?;
    )?;
//

//    // need llvm instead of arith to compare pointers
    // need llvm instead of arith to compare pointers
//    let is_null_ptr = entry.append_op_result(
    let is_null_ptr = entry.append_op_result(
//        OperationBuilder::new("llvm.icmp", location)
        OperationBuilder::new("llvm.icmp", location)
//            .add_operands(&[result_ptr, null_ptr])
            .add_operands(&[result_ptr, null_ptr])
//            .add_attributes(&[(
            .add_attributes(&[(
//                Identifier::new(context, "predicate"),
                Identifier::new(context, "predicate"),
//                IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
//            )])
            )])
//            .add_results(&[IntegerType::new(context, 1).into()])
            .add_results(&[IntegerType::new(context, 1).into()])
//            .build()?,
            .build()?,
//    )?;
    )?;
//

//    let block_is_null = helper.append_block(Block::new(&[]));
    let block_is_null = helper.append_block(Block::new(&[]));
//    let block_is_found = helper.append_block(Block::new(&[]));
    let block_is_found = helper.append_block(Block::new(&[]));
//    let block_final = helper.append_block(Block::new(&[
    let block_final = helper.append_block(Block::new(&[
//        (llvm::r#type::pointer(context, 0), location),
        (llvm::r#type::pointer(context, 0), location),
//        (value_ty, location),
        (value_ty, location),
//    ]));
    ]));
//

//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        is_null_ptr,
        is_null_ptr,
//        block_is_null,
        block_is_null,
//        block_is_found,
        block_is_found,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    // null block
    // null block
//    {
    {
//        let alloc_size = block_is_null.const_int(context, location, value_layout.size(), 64)?;
        let alloc_size = block_is_null.const_int(context, location, value_layout.size(), 64)?;
//

//        let value_ptr = block_is_null.append_op_result(ReallocBindingsMeta::realloc(
        let value_ptr = block_is_null.append_op_result(ReallocBindingsMeta::realloc(
//            context, result_ptr, alloc_size, location,
            context, result_ptr, alloc_size, location,
//        ))?;
        ))?;
//

//        let default_value = registry
        let default_value = registry
//            .get_type(&info.branch_signatures()[0].vars[1].ty)?
            .get_type(&info.branch_signatures()[0].vars[1].ty)?
//            .build_default(
            .build_default(
//                context,
                context,
//                registry,
                registry,
//                block_is_null,
                block_is_null,
//                location,
                location,
//                helper,
                helper,
//                metadata,
                metadata,
//                &info.branch_signatures()[0].vars[1].ty,
                &info.branch_signatures()[0].vars[1].ty,
//            )?;
            )?;
//

//        block_is_null.append_operation(cf::br(block_final, &[value_ptr, default_value], location));
        block_is_null.append_operation(cf::br(block_final, &[value_ptr, default_value], location));
//    }
    }
//

//    // found block
    // found block
//    {
    {
//        let loaded_val_ptr = block_is_found.load(
        let loaded_val_ptr = block_is_found.load(
//            context,
            context,
//            location,
            location,
//            result_ptr,
            result_ptr,
//            value_ty,
            value_ty,
//            Some(value_layout.align()),
            Some(value_layout.align()),
//        )?;
        )?;
//        block_is_found.append_operation(cf::br(
        block_is_found.append_operation(cf::br(
//            block_final,
            block_final,
//            &[result_ptr, loaded_val_ptr],
            &[result_ptr, loaded_val_ptr],
//            location,
            location,
//        ));
        ));
//    }
    }
//

//    // construct the struct
    // construct the struct
//

//    let entry_value = block_final.append_op_result(llvm::undef(entry_ty, location))?;
    let entry_value = block_final.append_op_result(llvm::undef(entry_ty, location))?;
//

//    let value_ptr = block_final.argument(0)?.into();
    let value_ptr = block_final.argument(0)?.into();
//    let value = block_final.argument(1)?.into();
    let value = block_final.argument(1)?.into();
//

//    let entry_value = block_final.insert_value(context, location, entry_value, key_value, 0)?;
    let entry_value = block_final.insert_value(context, location, entry_value, key_value, 0)?;
//

//    let entry_value = block_final.insert_value(context, location, entry_value, value_ptr, 1)?;
    let entry_value = block_final.insert_value(context, location, entry_value, value_ptr, 1)?;
//

//    let entry_value = block_final.insert_value(context, location, entry_value, dict_ptr, 2)?;
    let entry_value = block_final.insert_value(context, location, entry_value, dict_ptr, 2)?;
//

//    block_final.append_operation(helper.br(0, &[entry_value, value], location));
    block_final.append_operation(helper.br(0, &[entry_value, value], location));
//

//    Ok(())
    Ok(())
//}
}
//

//pub fn build_finalize<'ctx, 'this>(
pub fn build_finalize<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let value_type = registry.get_type(&info.param_signatures()[1].ty)?;
    let value_type = registry.get_type(&info.param_signatures()[1].ty)?;
//    let value_layout = value_type.layout(registry)?;
    let value_layout = value_type.layout(registry)?;
//

//    let key_ty = IntegerType::new(context, 252).into();
    let key_ty = IntegerType::new(context, 252).into();
//    let key_layout = get_integer_layout(252);
    let key_layout = get_integer_layout(252);
//

//    let entry_value = entry.argument(0)?.into();
    let entry_value = entry.argument(0)?.into();
//    let new_value = entry.argument(1)?.into();
    let new_value = entry.argument(1)?.into();
//

//    let key_value = entry.extract_value(context, location, entry_value, key_ty, 0)?;
    let key_value = entry.extract_value(context, location, entry_value, key_ty, 0)?;
//

//    let value_ptr = entry.extract_value(
    let value_ptr = entry.extract_value(
//        context,
        context,
//        location,
        location,
//        entry_value,
        entry_value,
//        llvm::r#type::pointer(context, 0),
        llvm::r#type::pointer(context, 0),
//        1,
        1,
//    )?;
    )?;
//

//    let dict_ptr = entry.extract_value(
    let dict_ptr = entry.extract_value(
//        context,
        context,
//        location,
        location,
//        entry_value,
        entry_value,
//        llvm::r#type::pointer(context, 0),
        llvm::r#type::pointer(context, 0),
//        2,
        2,
//    )?;
    )?;
//

//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        value_ptr,
        value_ptr,
//        new_value,
        new_value,
//        Some(value_layout.align()),
        Some(value_layout.align()),
//    );
    );
//

//    let key_ptr =
    let key_ptr =
//        helper
        helper
//            .init_block()
            .init_block()
//            .alloca1(context, location, key_ty, Some(key_layout.align()))?;
            .alloca1(context, location, key_ty, Some(key_layout.align()))?;
//

//    entry.store(
    entry.store(
//        context,
        context,
//        location,
        location,
//        key_ptr,
        key_ptr,
//        key_value,
        key_value,
//        Some(key_layout.align()),
        Some(key_layout.align()),
//    );
    );
//

//    // call insert
    // call insert
//

//    let runtime_bindings = metadata
    let runtime_bindings = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .expect("Runtime library not available.");
        .expect("Runtime library not available.");
//

//    runtime_bindings.dict_insert(
    runtime_bindings.dict_insert(
//        context, helper, entry, dict_ptr, key_ptr, value_ptr, location,
        context, helper, entry, dict_ptr, key_ptr, value_ptr, location,
//    )?;
    )?;
//

//    entry.append_operation(helper.br(0, &[dict_ptr], location));
    entry.append_operation(helper.br(0, &[dict_ptr], location));
//

//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::utils::test::{jit_dict, load_cairo, run_program_assert_output};
    use crate::utils::test::{jit_dict, load_cairo, run_program_assert_output};
//

//    #[test]
    #[test]
//    fn run_dict_insert() {
    fn run_dict_insert() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::Default;
            use traits::Default;
//            use dict::Felt252DictTrait;
            use dict::Felt252DictTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut dict: Felt252Dict<u32> = Default::default();
                let mut dict: Felt252Dict<u32> = Default::default();
//                dict.insert(2, 1_u32);
                dict.insert(2, 1_u32);
//                dict.get(2)
                dict.get(2)
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], 1u32.into());
        run_program_assert_output(&program, "run_test", &[], 1u32.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn run_dict_insert_big() {
    fn run_dict_insert_big() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::Default;
            use traits::Default;
//            use dict::Felt252DictTrait;
            use dict::Felt252DictTrait;
//

//            fn run_test() -> u64 {
            fn run_test() -> u64 {
//                let mut dict: Felt252Dict<u64> = Default::default();
                let mut dict: Felt252Dict<u64> = Default::default();
//                dict.insert(200000000, 4_u64);
                dict.insert(200000000, 4_u64);
//                dict.get(200000000)
                dict.get(200000000)
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], 4u64.into());
        run_program_assert_output(&program, "run_test", &[], 4u64.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn run_dict_insert_ret_dict() {
    fn run_dict_insert_ret_dict() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::Default;
            use traits::Default;
//            use dict::Felt252DictTrait;
            use dict::Felt252DictTrait;
//

//            fn run_test() -> Felt252Dict<u32> {
            fn run_test() -> Felt252Dict<u32> {
//                let mut dict: Felt252Dict<u32> = Default::default();
                let mut dict: Felt252Dict<u32> = Default::default();
//                dict.insert(2, 1_u32);
                dict.insert(2, 1_u32);
//                dict
                dict
//            }
            }
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            jit_dict!(
            jit_dict!(
//                2 => 1u32
                2 => 1u32
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn run_dict_insert_multiple() {
    fn run_dict_insert_multiple() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::Default;
            use traits::Default;
//            use dict::Felt252DictTrait;
            use dict::Felt252DictTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut dict: Felt252Dict<u32> = Default::default();
                let mut dict: Felt252Dict<u32> = Default::default();
//                dict.insert(2, 1_u32);
                dict.insert(2, 1_u32);
//                dict.insert(3, 1_u32);
                dict.insert(3, 1_u32);
//                dict.insert(4, 1_u32);
                dict.insert(4, 1_u32);
//                dict.insert(5, 1_u32);
                dict.insert(5, 1_u32);
//                dict.insert(6, 1_u32);
                dict.insert(6, 1_u32);
//                dict.insert(7, 1_u32);
                dict.insert(7, 1_u32);
//                dict.insert(8, 1_u32);
                dict.insert(8, 1_u32);
//                dict.insert(9, 1_u32);
                dict.insert(9, 1_u32);
//                dict.insert(10, 1_u32);
                dict.insert(10, 1_u32);
//                dict.insert(11, 1_u32);
                dict.insert(11, 1_u32);
//                dict.insert(12, 1_u32);
                dict.insert(12, 1_u32);
//                dict.insert(13, 1_u32);
                dict.insert(13, 1_u32);
//                dict.insert(14, 1_u32);
                dict.insert(14, 1_u32);
//                dict.insert(15, 1_u32);
                dict.insert(15, 1_u32);
//                dict.insert(16, 1_u32);
                dict.insert(16, 1_u32);
//                dict.insert(17, 1_u32);
                dict.insert(17, 1_u32);
//                dict.insert(18, 1345432_u32);
                dict.insert(18, 1345432_u32);
//                dict.get(18)
                dict.get(18)
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], 1345432_u32.into());
        run_program_assert_output(&program, "run_test", &[], 1345432_u32.into());
//    }
    }
//}
}
