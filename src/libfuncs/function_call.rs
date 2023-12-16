//! # Function call libfuncs
//!
//! Includes logic for handling direct tail recursive function calls. More information on this topic
//! at the [tail recursive metadata](crate::metadata::tail_recursion).

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{tail_recursion::TailRecursionMeta, MetadataStorage},
    types::TypeBuilder,
    utils::generate_function_name,
};
use cairo_lang_sierra::{
    extensions::{function_call::FunctionCallConcreteLibfunc, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith, cf, func, index,
        llvm::{self, AllocaOptions},
        memref,
    },
    ir::{
        attribute::{FlatSymbolRefAttribute, IntegerAttribute, TypeAttribute},
        r#type::IntegerType,
        Block, Location, Type,
    },
    Context,
};

/// Generate MLIR operations for the `function_call` libfunc.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &FunctionCallConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let mut arguments = Vec::new();
    let mut result_types = Vec::new();

    for (idx, type_id) in info.function.signature.param_types.iter().enumerate() {
        let type_info = registry.get_type(type_id)?;

        if !(type_info.is_builtin() && type_info.is_zst(registry)) {
            arguments.push(entry.argument(idx)?.into());
        }
    }

    let mut num_return_ptrs = 0;
    for type_id in &info.function.signature.ret_types {
        let type_info = registry.get_type(type_id)?;

        if !(type_info.is_builtin() && type_info.is_zst(registry)) {
            let ty = type_info.build(context, helper, registry, metadata, type_id)?;

            if type_info.is_memory_allocated(registry) {
                let layout = type_info.layout(registry)?;

                let k1 = helper
                    .init_block()
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let stack_ptr = helper
                    .init_block()
                    .append_operation(llvm::alloca(
                        context,
                        k1,
                        llvm::r#type::opaque_pointer(context),
                        location,
                        AllocaOptions::new()
                            .align(Some(IntegerAttribute::new(
                                layout.align() as i64,
                                IntegerType::new(context, 64).into(),
                            )))
                            .elem_type(Some(TypeAttribute::new(ty))),
                    ))
                    .result(0)?
                    .into();

                arguments.insert(0, stack_ptr);
                num_return_ptrs += 1;
            } else {
                result_types.push(ty);
            }
        }
    }

    if let Some(tailrec_meta) = metadata.get_mut::<TailRecursionMeta>() {
        let op0 = entry.append_operation(memref::load(tailrec_meta.depth_counter(), &[], location));
        let op1 = entry.append_operation(index::constant(
            context,
            IntegerAttribute::new(1, Type::index(context)),
            location,
        ));
        let op2 = entry.append_operation(index::add(
            op0.result(0)?.into(),
            op1.result(0)?.into(),
            location,
        ));
        entry.append_operation(memref::store(
            op2.result(0)?.into(),
            tailrec_meta.depth_counter(),
            &[],
            location,
        ));

        entry.append_operation(cf::br(
            &tailrec_meta.recursion_target(),
            &arguments
                .iter()
                .skip(num_return_ptrs)
                .copied()
                .collect::<Vec<_>>(),
            location,
        ));

        let cont_block = helper.append_block(Block::new(
            &result_types
                .iter()
                .copied()
                .map(|ty| (ty, location))
                .collect::<Vec<_>>(),
        ));
        tailrec_meta.set_return_target(cont_block);

        let mut results = Vec::new();
        let mut count = 0;
        let mut iter = arguments.iter().copied();
        for var_info in &info.signature.branch_signatures[0].vars {
            let type_info = registry.get_type(&var_info.ty)?;

            results.push(if type_info.is_builtin() && type_info.is_zst(registry) {
                entry
                    .append_operation(llvm::undef(
                        type_info.build(context, helper, registry, metadata, &var_info.ty)?,
                        location,
                    ))
                    .result(0)?
                    .into()
            } else if type_info.is_memory_allocated(registry) {
                iter.next().unwrap()
            } else {
                let value = cont_block.argument(count)?.into();
                count += 1;
                value
            });
        }

        entry.append_operation(helper.br(0, &results, location));
    } else {
        let op0 = entry.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, &generate_function_name(&info.function.id)),
            &arguments,
            &result_types,
            location,
        ));

        let mut results = Vec::new();
        let mut count = 0;
        let mut iter = arguments.iter().copied();
        for var_info in &info.signature.branch_signatures[0].vars {
            let type_info = registry.get_type(&var_info.ty)?;

            results.push(if type_info.is_builtin() && type_info.is_zst(registry) {
                entry
                    .append_operation(llvm::undef(
                        type_info.build(context, helper, registry, metadata, &var_info.ty)?,
                        location,
                    ))
                    .result(0)?
                    .into()
            } else if type_info.is_memory_allocated(registry) {
                iter.next().unwrap()
            } else {
                let value = op0.result(count)?.into();
                count += 1;
                value
            });
        }

        entry.append_operation(helper.br(0, &results, location));
    }

    Ok(())
}
