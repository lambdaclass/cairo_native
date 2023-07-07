//! # Pedersen hashing libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        lib_func::SignatureOnlyConcreteLibfunc, pedersen::PedersenConcreteLibfunc,
        structure::StructConcreteLibfunc, ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, LoadStoreOptions},
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location, Value,
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
    selector: &PedersenConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        PedersenConcreteLibfunc::PedersenHash(info) => {
            build_pedersen(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_pedersen<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, '_>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    {
        metadata
            .get_mut::<RuntimeBindingsMeta>()
            .expect("Runtime library not available.");
    }

    let felt252_ty = registry
        .get_type(&info.param_signatures()[1].ty)?
        .build(context, helper, registry, metadata)?;
    let layout = registry
        .get_type(&info.param_signatures()[1].ty)?
        .layout(registry)?;

    let pedersen_builtin = entry.argument(0)?.into();
    let lhs = entry.argument(1)?.into();
    let rhs = entry.argument(2)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
        location,
    ));
    let const_1 = op.result(0)?.into();

    let op = entry.append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    layout.align().try_into()?,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[const_1])
            .add_results(&[llvm::r#type::pointer(felt252_ty, 0)])
            .build(),
    );
    let lhs_ptr = op.result(0)?.into();

    let op = entry.append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    layout.align().try_into()?,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[const_1])
            .add_results(&[llvm::r#type::pointer(felt252_ty, 0)])
            .build(),
    );
    let rhs_ptr = op.result(0)?.into();

    let op = entry.append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    layout.align().try_into()?,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[const_1])
            .add_results(&[llvm::r#type::pointer(felt252_ty, 0)])
            .build(),
    );
    let dst_ptr = op.result(0)?.into();

    entry.append_operation(llvm::store(
        context,
        lhs,
        lhs_ptr,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    entry.append_operation(llvm::store(
        context,
        rhs,
        rhs_ptr,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    runtime_bindings
        .libfunc_pedersen(context, helper, entry, dst_ptr, lhs_ptr, rhs_ptr, location)?;

    let op = entry.append_operation(llvm::load(
        context,
        dst_ptr,
        felt252_ty,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            layout.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    let result = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[pedersen_builtin, result], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        libfuncs::felt252::test::f,
        utils::test::{load_cairo, run_program},
    };
    use serde_json::json;

    pub fn true_js() -> serde_json::Value {
        json!([1, []])
    }

    pub fn false_js() -> serde_json::Value {
        json!([0, []])
    }

    #[test]
    fn run_pedersen() {
        let program = load_cairo!(
            use hash::pedersen;

            fn run_test(a: felt252, b: felt252) -> felt252 {
                pedersen(a, b)
            }
        );

        let result = run_program(&program, "run_test", json!([(), f("1"), f("1")]));
        assert_eq!(result, json!([f("1")]));

        let result = run_program(&program, "run_test", json!([false_js()]));
        assert_eq!(result, json!([true_js()]));
    }
}
