////! # Debug libfuncs
//! # Debug libfuncs
//

//// Printable: 9-13, 27, 32, 33-126
// Printable: 9-13, 27, 32, 33-126
////     is_ascii_graphic() -> 33-126
//     is_ascii_graphic() -> 33-126
////     is_ascii_whitespace():
//     is_ascii_whitespace():
////         U+0009 HORIZONTAL TAB
//         U+0009 HORIZONTAL TAB
////         U+000A LINE FEED
//         U+000A LINE FEED
////         U+000C FORM FEED
//         U+000C FORM FEED
////         U+000D CARRIAGE RETURN.
//         U+000D CARRIAGE RETURN.
////         U+0020 SPACE
//         U+0020 SPACE
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    error::Result,
    error::Result,
//    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        debug::DebugConcreteLibfunc,
        debug::DebugConcreteLibfunc,
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{arith, cf, llvm},
    dialect::{arith, cf, llvm},
//    ir::{
    ir::{
//        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
//        r#type::IntegerType,
        r#type::IntegerType,
//        Block, Location,
        Block, Location,
//    },
    },
//    Context,
    Context,
//};
};
//

//pub fn build<'ctx>(
pub fn build<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &Block<'ctx>,
    entry: &Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, '_>,
    helper: &LibfuncHelper<'ctx, '_>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &DebugConcreteLibfunc,
    selector: &DebugConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        DebugConcreteLibfunc::Print(info) => {
        DebugConcreteLibfunc::Print(info) => {
//            build_print(context, registry, entry, location, helper, metadata, info)
            build_print(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

//pub fn build_print<'ctx>(
pub fn build_print<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &Block<'ctx>,
    entry: &Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, '_>,
    helper: &LibfuncHelper<'ctx, '_>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let stdout_fd = entry
    let stdout_fd = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 32).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 32).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let values_ptr = entry
    let values_ptr = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            entry.argument(0)?.into(),
            entry.argument(0)?.into(),
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let values_start = entry
    let values_start = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            entry.argument(0)?.into(),
            entry.argument(0)?.into(),
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            IntegerType::new(context, 32).into(),
            IntegerType::new(context, 32).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let values_end = entry
    let values_end = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            entry.argument(0)?.into(),
            entry.argument(0)?.into(),
//            DenseI64ArrayAttribute::new(context, &[2]),
            DenseI64ArrayAttribute::new(context, &[2]),
//            IntegerType::new(context, 32).into(),
            IntegerType::new(context, 32).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let runtime_bindings = metadata
    let runtime_bindings = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .expect("Runtime library not available.");
        .expect("Runtime library not available.");
//

//    let values_len = entry
    let values_len = entry
//        .append_operation(arith::subi(values_end, values_start, location))
        .append_operation(arith::subi(values_end, values_start, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let values_ptr = {
    let values_ptr = {
//        let values_start = entry
        let values_start = entry
//            .append_operation(arith::extui(
            .append_operation(arith::extui(
//                values_start,
                values_start,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        entry
        entry
//            .append_operation(llvm::get_element_ptr_dynamic(
            .append_operation(llvm::get_element_ptr_dynamic(
//                context,
                context,
//                values_ptr,
                values_ptr,
//                &[values_start],
                &[values_start],
//                IntegerType::new(context, 252).into(),
                IntegerType::new(context, 252).into(),
//                llvm::r#type::pointer(context, 0),
                llvm::r#type::pointer(context, 0),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into()
            .into()
//    };
    };
//

//    let return_code = runtime_bindings.libfunc_debug_print(
    let return_code = runtime_bindings.libfunc_debug_print(
//        context, helper, entry, stdout_fd, values_ptr, values_len, location,
        context, helper, entry, stdout_fd, values_ptr, values_len, location,
//    )?;
    )?;
//

//    let k0 = entry
    let k0 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 32).into(), 0).into(),
            IntegerAttribute::new(IntegerType::new(context, 32).into(), 0).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let return_code_is_ok = entry
    let return_code_is_ok = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            arith::CmpiPredicate::Eq,
            arith::CmpiPredicate::Eq,
//            return_code,
            return_code,
//            k0,
            k0,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    cf::assert(
    cf::assert(
//        context,
        context,
//        return_code_is_ok,
        return_code_is_ok,
//        "Print libfunc invocation failed.",
        "Print libfunc invocation failed.",
//        location,
        location,
//    );
    );
//

//    entry.append_operation(helper.br(0, &[], location));
    entry.append_operation(helper.br(0, &[], location));
//

//    Ok(())
    Ok(())
//}
}
