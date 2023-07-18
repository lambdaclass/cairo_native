//! # Casting libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        casts::CastConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith,
    ir::{r#type::IntegerType, Block, Location},
    Context,
};

pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &CastConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        CastConcreteLibfunc::Downcast(_) => todo!(),
        CastConcreteLibfunc::Upcast(info) => {
            build_upcast(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_upcast<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let src_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let dst_ty = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;

    let src_width = src_ty.integer_width().unwrap();
    let dst_width = dst_ty.integer_width().unwrap();
    assert!(src_width <= dst_width);

    let result = if src_width == dst_width {
        entry.argument(0)?.into()
    } else {
        entry
            .append_operation(arith::extui(
                entry.argument(0)?.into(),
                IntegerType::new(context, dst_width.try_into()?).into(),
                location,
            ))
            .result(0)?
            .into()
    };

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{load_cairo, run_program};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use serde_json::json;

    lazy_static! {
        static ref UPCAST: (String, Program) = load_cairo! {
            use core::integer::upcast;

            fn run_test(v8: u8, v16: u16, v32: u32, v64: u64, v128: u128) -> (
                (u8,),
                (u16, u16),
                (u32, u32, u32),
                (u64, u64, u64, u64),
                (u128, u128, u128, u128, u128)
            ) {
                (
                    (upcast(v8),),
                    (upcast(v8), upcast(v16)),
                    (upcast(v8), upcast(v16), upcast(v32)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128)),
                )
            }
        };
    }

    #[test]
    fn upcast() {
        assert_eq!(
            run_program(
                &UPCAST,
                "run_test",
                json!([
                    0xFFu8,
                    0xFFFFu16,
                    0xFFFFFFFFu32,
                    0xFFFFFFFFFFFFFFFFu64,
                    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFu128
                ])
            ),
            json!([[
                [0xFFu8],
                [0xFFu8, 0xFFFFu16],
                [0xFFu8, 0xFFFFu16, 0xFFFFFFFFu32],
                [0xFFu8, 0xFFFFu16, 0xFFFFFFFFu32, 0xFFFFFFFFFFFFFFFFu64],
                [
                    0xFFu8,
                    0xFFFFu16,
                    0xFFFFFFFFu32,
                    0xFFFFFFFFFFFFFFFFu64,
                    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFu128
                ],
            ]])
        );
    }
}
