////! # `Felt` dictionary libfuncs
//! # `Felt` dictionary libfuncs
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
//        felt252_dict::Felt252DictConcreteLibfunc,
        felt252_dict::Felt252DictConcreteLibfunc,
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
//    dialect::arith,
    dialect::arith,
//    ir::{r#type::IntegerType, Block, Location},
    ir::{r#type::IntegerType, Block, Location},
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
//    selector: &Felt252DictConcreteLibfunc,
    selector: &Felt252DictConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        Felt252DictConcreteLibfunc::New(info) => {
        Felt252DictConcreteLibfunc::New(info) => {
//            build_new(context, registry, entry, location, helper, metadata, info)
            build_new(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Felt252DictConcreteLibfunc::Squash(info) => {
        Felt252DictConcreteLibfunc::Squash(info) => {
//            build_squash(context, registry, entry, location, helper, metadata, info)
            build_squash(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

//pub fn build_new<'ctx, 'this>(
pub fn build_new<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let segment_arena =
    let segment_arena =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let runtime_bindings = metadata
    let runtime_bindings = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .expect("Runtime library not available.");
        .expect("Runtime library not available.");
//

//    let op = runtime_bindings.dict_alloc_new(context, helper, entry, location)?;
    let op = runtime_bindings.dict_alloc_new(context, helper, entry, location)?;
//    let dict_ptr = op.result(0)?.into();
    let dict_ptr = op.result(0)?.into();
//

//    entry.append_operation(helper.br(0, &[segment_arena, dict_ptr], location));
    entry.append_operation(helper.br(0, &[segment_arena, dict_ptr], location));
//    Ok(())
    Ok(())
//}
}
//

//pub fn build_squash<'ctx, 'this>(
pub fn build_squash<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//    let gas_builtin = entry.argument(1)?.into();
    let gas_builtin = entry.argument(1)?.into();
//    let segment_arena =
    let segment_arena =
//        super::increment_builtin_counter(context, entry, location, entry.argument(2)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(2)?.into())?;
//    let dict_ptr = entry.argument(3)?.into();
    let dict_ptr = entry.argument(3)?.into();
//

//    let runtime_bindings = metadata
    let runtime_bindings = metadata
//        .get_mut::<RuntimeBindingsMeta>()
        .get_mut::<RuntimeBindingsMeta>()
//        .expect("Runtime library not available.");
        .expect("Runtime library not available.");
//

//    let gas_refund = runtime_bindings
    let gas_refund = runtime_bindings
//        .dict_gas_refund(context, helper, entry, dict_ptr, location)?
        .dict_gas_refund(context, helper, entry, dict_ptr, location)?
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let gas_refund = entry
    let gas_refund = entry
//        .append_operation(arith::extui(
        .append_operation(arith::extui(
//            gas_refund,
            gas_refund,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let new_gas_builtin = entry
    let new_gas_builtin = entry
//        .append_operation(arith::addi(gas_builtin, gas_refund, location))
        .append_operation(arith::addi(gas_builtin, gas_refund, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.br(
    entry.append_operation(helper.br(
//        0,
        0,
//        &[
        &[
//            range_check,
            range_check,
//            new_gas_builtin,
            new_gas_builtin,
//            segment_arena,
            segment_arena,
//            entry.argument(3)?.into(),
            entry.argument(3)?.into(),
//        ],
        ],
//        location,
        location,
//    ));
    ));
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
//    use crate::utils::test::{jit_dict, jit_struct, load_cairo, run_program_assert_output};
    use crate::utils::test::{jit_dict, jit_struct, load_cairo, run_program_assert_output};
//

//    #[test]
    #[test]
//    fn run_dict_new() {
    fn run_dict_new() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::Default;
            use traits::Default;
//            use dict::Felt252DictTrait;
            use dict::Felt252DictTrait;
//

//            fn run_test() {
            fn run_test() {
//                let mut _dict: Felt252Dict<u32> = Default::default();
                let mut _dict: Felt252Dict<u32> = Default::default();
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], jit_struct!());
        run_program_assert_output(&program, "run_test", &[], jit_struct!());
//    }
    }
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
//                dict.insert(1, 2_u32);
                dict.insert(1, 2_u32);
//                dict.insert(2, 3_u32);
                dict.insert(2, 3_u32);
//                dict.insert(3, 4_u32);
                dict.insert(3, 4_u32);
//                dict.insert(4, 5_u32);
                dict.insert(4, 5_u32);
//                dict.insert(5, 6_u32);
                dict.insert(5, 6_u32);
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
//                1 => 2u32,
                1 => 2u32,
//                2 => 3u32,
                2 => 3u32,
//                3 => 4u32,
                3 => 4u32,
//                4 => 5u32,
                4 => 5u32,
//                5 => 6u32,
                5 => 6u32,
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn run_dict_deserialize() {
    fn run_dict_deserialize() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::Default;
            use traits::Default;
//            use dict::Felt252DictTrait;
            use dict::Felt252DictTrait;
//

//            fn run_test(mut dict: Felt252Dict<u32>) -> Felt252Dict<u32> {
            fn run_test(mut dict: Felt252Dict<u32>) -> Felt252Dict<u32> {
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
//            &[jit_dict!(
            &[jit_dict!(
//                1 => 2u32,
                1 => 2u32,
//                2 => 3u32,
                2 => 3u32,
//                3 => 4u32,
                3 => 4u32,
//                4 => 5u32,
                4 => 5u32,
//                5 => 6u32,
                5 => 6u32,
//            )],
            )],
//            jit_dict!(
            jit_dict!(
//                1 => 2u32,
                1 => 2u32,
//                2 => 3u32,
                2 => 3u32,
//                3 => 4u32,
                3 => 4u32,
//                4 => 5u32,
                4 => 5u32,
//                5 => 6u32,
                5 => 6u32,
//            ),
            ),
//        );
        );
//    }
    }
//}
}
