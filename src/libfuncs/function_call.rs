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
    dialect::{cf, func, index, memref},
    ir::{
        attribute::{FlatSymbolRefAttribute, IntegerAttribute},
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
    let arguments = (0..entry.argument_count())
        .map(|i| Result::Ok(entry.argument(i)?.into()))
        .collect::<Result<Vec<_>>>()?;

    let result_types = info.signature.branch_signatures[0]
        .vars
        .iter()
        .map(|x| Result::Ok(registry.build_type(context, helper, registry, metadata, &x.ty)?))
        .collect::<Result<Vec<_>>>()?;

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
            &arguments,
            location,
        ));

        let cont_block = helper.append_block(Block::new(
            &result_types
                .iter()
                .copied()
                .map(|ty| (ty, location))
                .collect::<Vec<_>>(),
        ));

        cont_block.append_operation(
            helper.br(
                0,
                &(0..result_types.len())
                    .map(|i| Result::Ok(cont_block.argument(i)?.into()))
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

        entry.append_operation(
            helper.br(
                0,
                &result_types
                    .iter()
                    .enumerate()
                    .map(|(i, _)| Result::Ok(op0.result(i)?.into()))
                    .collect::<Result<Vec<_>>>()?,
                location,
            ),
        );
    }

    Ok(())
}
