use super::{LibfuncBuilder, LibfuncHelper};
use crate::{generate_function_name, metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{function_call::FunctionCallConcreteLibfunc, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, func},
    ir::{
        attribute::{FlatSymbolRefAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location, Value,
    },
    Context,
};

pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &FunctionCallConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let mut arguments = (0..entry.argument_count())
        .map(|i| entry.argument(i).unwrap().into())
        .collect::<Vec<_>>();
    let mut result_types = info.signature.branch_signatures[0]
        .vars
        .iter()
        .map(|x| {
            registry
                .get_type(&x.ty)
                .unwrap()
                .build(context, helper, registry, metadata)
                .unwrap()
        })
        .collect::<Vec<_>>();

    // Avoid returning locals on the stack, aka. dangling pointers.
    let mut transfer_types = Vec::new();
    for ret_ty in &info.function.signature.ret_types {
        if registry.get_type(ret_ty).unwrap().variants().is_some() {
            transfer_types.push(
                registry
                    .get_type(ret_ty)
                    .unwrap()
                    .build(context, helper, registry, metadata)
                    .unwrap(),
            );
        }
    }

    let op0 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
        location,
    ));

    let mut transfer_values = Vec::new();
    arguments.extend(
        result_types
            .extract_if(|ty| transfer_types.contains(ty))
            .map(|ty| {
                let align = crate::ffi::get_abi_alignment(
                    helper,
                    &crate::ffi::get_pointer_element_type(&ty),
                );

                let transfer_value: Value = entry
                    .append_operation(
                        OperationBuilder::new("llvm.alloca", location)
                            .add_attributes(&[(
                                Identifier::new(context, "alignment"),
                                IntegerAttribute::new(
                                    align.try_into().unwrap(),
                                    IntegerType::new(context, 64).into(),
                                )
                                .into(),
                            )])
                            .add_operands(&[op0.result(0).unwrap().into()])
                            .add_results(&[ty])
                            .build(),
                    )
                    .result(0)
                    .unwrap()
                    .into();

                transfer_values.push(transfer_value);
                transfer_value
            }),
    );

    let op1 = entry.append_operation(func::call(
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
                .map(|(i, _)| op1.result(i).unwrap().into())
                .chain(transfer_values)
                .collect::<Vec<_>>(),
            location,
        ),
    );

    Ok(())
}
