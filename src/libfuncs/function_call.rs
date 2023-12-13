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
    utils::{generate_function_name, ProgramRegistryExt},
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
    let mut arguments = (0..entry.argument_count())
        .map(|i| Result::Ok(entry.argument(i)?.into()))
        .collect::<Result<Vec<_>>>()?;

    let mut result_types = info.signature.branch_signatures[0]
        .vars
        .iter()
        .map(|x| Result::Ok(registry.build_type(context, helper, registry, metadata, &x.ty)?))
        .collect::<Result<Vec<_>>>()?;

    // Allocate space for memory-allocated return types and move them from `result_types` into
    // `arguments`.
    let mut num_return_ptrs = 0;
    for (idx, var_info) in info.signature.branch_signatures[0]
        .vars
        .iter()
        .enumerate()
        .rev()
    {
        let type_info = registry.get_type(&var_info.ty).unwrap();
        if type_info.is_memory_allocated(registry) {
            let layout = type_info.layout(registry)?;
            let ty = result_types.remove(idx);

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

        let mut iter = arguments.iter().copied();
        let mut arg_index = 0;
        cont_block.append_operation(
            helper.br(
                0,
                &info.signature.branch_signatures[0]
                    .vars
                    .iter()
                    .map(|var_info| {
                        let type_info = registry.get_type(&var_info.ty)?;
                        Result::Ok(if type_info.is_memory_allocated(registry) {
                            iter.next().unwrap()
                        } else {
                            let value = cont_block.argument(arg_index)?.into();
                            arg_index += 1;
                            value
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                location,
            ),
        );

        tailrec_meta.set_return_target(cont_block);
    } else {
        let op0 = entry.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, &generate_function_name(&info.function.id)),
            &arguments,
            &result_types,
            location,
        ));

        let mut iter = arguments.iter().copied();
        entry.append_operation(
            helper.br(
                0,
                &info.signature.branch_signatures[0]
                    .vars
                    .iter()
                    .enumerate()
                    .map(|(i, var_info)| {
                        let type_info = registry.get_type(&var_info.ty)?;
                        Result::Ok(if type_info.is_memory_allocated(registry) {
                            iter.next().unwrap()
                        } else {
                            op0.result(i)?.into()
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                location,
            ),
        );
    }

    Ok(())
}
