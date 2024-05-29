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

    let selector = helper
        .init_block()
        .const_int(context, location, info.selector.clone(), 256)?;
    let selector_ptr =
        helper
            .init_block()
            .alloca1(context, location, llvm::r#type::pointer(context, 0), None)?;
    helper
        .init_block()
        .store(context, location, selector_ptr, selector, None);

    let args_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        None,
    )?;
    entry.store(context, location, args_ptr, entry.argument(0)?.into(), None);

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.")
        .vtable_callback(
            context,
            helper,
            entry,
            location,
            result_ptr,
            selector_ptr,
            args_ptr,
        )?;

    let result = entry.append_op_result(llvm::load(
        context,
        result_ptr,
        result_type,
        location,
        LoadStoreOptions::new(),
    ))?;

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}
