use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        starknet::testing::CheatcodeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm::{self, alloca, AllocaOptions, LoadStoreOptions},
    ir::{
        attribute::{IntegerAttribute, TypeAttribute},
        r#type::IntegerType,
        Block, Location,
    },
    Context,
};

use crate::{
    block_ext::BlockExt,
    error::Result,
    libfuncs::LibfuncHelper,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::ProgramRegistryExt,
};

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &CheatcodeConcreteLibfunc,
) -> Result<()> {
    // calculate the result layout and type
    let (result_type, result_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    dbg!(result_type);

    // allocate the result pointer
    let result_ptr = helper
        .init_block()
        .append_operation(alloca(
            context,
            helper.init_block().const_int(context, location, 1, 64)?,
            llvm::r#type::pointer(context, 0),
            location,
            AllocaOptions::new()
                .align(Some(IntegerAttribute::new(
                    IntegerType::new(context, 64).into(),
                    result_layout.align().try_into()?,
                )))
                .elem_type(Some(TypeAttribute::new(result_type))),
        ))
        .result(0)?
        .into();

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.")
        .vtable_callback(context, helper, entry, location, result_ptr)?;

    let result = entry.append_op_result(llvm::load(
        context,
        result_ptr,
        result_type,
        location,
        LoadStoreOptions::new(),
    ))?;

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())

    // dbg!(result_ptr);

    // dbg!(&info.selector);

    // let selector_bytes = info.selector.to_bytes_be().1;
    // let selector = std::str::from_utf8(&selector_bytes).unwrap();

    // dbg!(selector);

    // let args_ptr = helper.init_block().alloca1(
    //     context,
    //     location,
    //     llvm::r#type::r#struct(
    //         context,
    //         &[llvm::r#type::r#struct(
    //             context,
    //             &[
    //                 llvm::r#type::pointer(context, 0),
    //                 IntegerType::new(context, 32).into(),
    //                 IntegerType::new(context, 32).into(),
    //                 IntegerType::new(context, 32).into(),
    //             ],
    //             false,
    //         )],
    //         false,
    //     ),
    //     Some(8),
    // )?;
    // entry.store(context, location, args_ptr, entry.argument(0)?.into(), None);

    // dbg!(args_ptr);

    // let ptr = entry
    //     .append_operation(llvm::load(
    //         context,
    //         entry.argument(1)?.into(),
    //         llvm::r#type::pointer(context, 0),
    //         location,
    //         LoadStoreOptions::default(),
    //     ))
    //     .result(0)?
    //     .into();

    // let fn_ptr_addr = entry
    //     .append_operation(llvm::get_element_ptr(
    //         context,
    //         entry.argument(1)?.into(),
    //         DenseI32ArrayAttribute::new(
    //             context,
    //             &[StarknetSyscallHandlerCallbacks::<()>::CHEATCODE.try_into()?],
    //         ),
    //         llvm::r#type::pointer(context, 0),
    //         llvm::r#type::pointer(context, 0),
    //         location,
    //     ))
    //     .result(0)?
    //     .into();
    // let fn_ptr = entry
    //     .append_operation(llvm::load(
    //         context,
    //         fn_ptr_addr,
    //         llvm::r#type::pointer(context, 0),
    //         location,
    //         LoadStoreOptions::default(),
    //     ))
    //     .result(0)?
    //     .into();

    // dbg!(fn_ptr);

    // entry.append_operation(
    //     OperationBuilder::new("llvm.call", location)
    //         .add_operands(&[
    //             fn_ptr, result_ptr, ptr, // gas_builtin_ptr,
    //             args_ptr,
    //         ])
    //         .build()?,
    // );

    // dbg!(result);

    // let result_tag = entry
    //     .append_operation(llvm::extract_value(
    //         context,
    //         result,
    //         DenseI64ArrayAttribute::new(context, &[0]),
    //         IntegerType::new(context, 1).into(),
    //         location,
    //     ))
    //     .result(0)?
    //     .into();

    // dbg!(result_tag);

    // let payload_err = {
    //     let ptr = entry
    //         .append_operation(
    //             OperationBuilder::new("llvm.getelementptr", location)
    //                 .add_attributes(&[
    //                     (
    //                         Identifier::new(context, "rawConstantIndices"),
    //                         DenseI32ArrayAttribute::new(
    //                             context,
    //                             &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
    //                         )
    //                         .into(),
    //                     ),
    //                     (
    //                         Identifier::new(context, "elem_type"),
    //                         TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
    //                     ),
    //                 ])
    //                 .add_operands(&[result_ptr])
    //                 .add_results(&[llvm::r#type::pointer(context, 0)])
    //                 .build()?,
    //         )
    //         .result(0)?
    //         .into();
    //     entry
    //         .append_operation(llvm::load(
    //             context,
    //             ptr,
    //             variant_tys[1].0,
    //             location,
    //             LoadStoreOptions::default(),
    //         ))
    //         .result(0)?
    //         .into()
    // };
}
