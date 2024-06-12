////! # Boolean libfuncs
//! # Boolean libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt, error::Result, metadata::MetadataStorage, types::TypeBuilder,
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, types::TypeBuilder,
//    utils::ProgramRegistryExt,
    utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        boolean::BoolConcreteLibfunc,
        boolean::BoolConcreteLibfunc,
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{arith, llvm},
    dialect::{arith, llvm},
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
//    selector: &BoolConcreteLibfunc,
    selector: &BoolConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        BoolConcreteLibfunc::And(info) => build_bool_binary(
        BoolConcreteLibfunc::And(info) => build_bool_binary(
//            context,
            context,
//            registry,
            registry,
//            entry,
            entry,
//            location,
            location,
//            helper,
            helper,
//            metadata,
            metadata,
//            info,
            info,
//            BoolOp::And,
            BoolOp::And,
//        ),
        ),
//        BoolConcreteLibfunc::Not(info) => {
        BoolConcreteLibfunc::Not(info) => {
//            build_bool_not(context, registry, entry, location, helper, metadata, info)
            build_bool_not(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        BoolConcreteLibfunc::Xor(info) => build_bool_binary(
        BoolConcreteLibfunc::Xor(info) => build_bool_binary(
//            context,
            context,
//            registry,
            registry,
//            entry,
            entry,
//            location,
            location,
//            helper,
            helper,
//            metadata,
            metadata,
//            info,
            info,
//            BoolOp::Xor,
            BoolOp::Xor,
//        ),
        ),
//        BoolConcreteLibfunc::Or(info) => build_bool_binary(
        BoolConcreteLibfunc::Or(info) => build_bool_binary(
//            context,
            context,
//            registry,
            registry,
//            entry,
            entry,
//            location,
            location,
//            helper,
            helper,
//            metadata,
            metadata,
//            info,
            info,
//            BoolOp::Or,
            BoolOp::Or,
//        ),
        ),
//        BoolConcreteLibfunc::ToFelt252(info) => {
        BoolConcreteLibfunc::ToFelt252(info) => {
//            build_bool_to_felt252(context, registry, entry, location, helper, metadata, info)
            build_bool_to_felt252(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

//#[derive(Debug, Clone, Copy)]
#[derive(Debug, Clone, Copy)]
//enum BoolOp {
enum BoolOp {
//    And,
    And,
//    Xor,
    Xor,
//    Or,
    Or,
//}
}
//

///// Generate MLIR operations for the `bool_not_impl` libfunc.
/// Generate MLIR operations for the `bool_not_impl` libfunc.
//#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
//fn build_bool_binary<'ctx, 'this>(
fn build_bool_binary<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//    bin_op: BoolOp,
    bin_op: BoolOp,
//) -> Result<()> {
) -> Result<()> {
//    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
//    let tag_bits = enum_ty
    let tag_bits = enum_ty
//        .variants()
        .variants()
//        .expect("bool is a enum and has variants")
        .expect("bool is a enum and has variants")
//        .len()
        .len()
//        .next_power_of_two()
        .next_power_of_two()
//        .trailing_zeros();
        .trailing_zeros();
//    let tag_ty = IntegerType::new(context, tag_bits).into();
    let tag_ty = IntegerType::new(context, tag_bits).into();
//

//    let lhs = entry.argument(0)?.into();
    let lhs = entry.argument(0)?.into();
//    let rhs = entry.argument(1)?.into();
    let rhs = entry.argument(1)?.into();
//

//    let lhs_tag = entry.extract_value(context, location, lhs, tag_ty, 0)?;
    let lhs_tag = entry.extract_value(context, location, lhs, tag_ty, 0)?;
//

//    let rhs_tag = entry.extract_value(context, location, rhs, tag_ty, 0)?;
    let rhs_tag = entry.extract_value(context, location, rhs, tag_ty, 0)?;
//

//    let new_tag_value = match bin_op {
    let new_tag_value = match bin_op {
//        BoolOp::And => entry.append_op_result(arith::andi(lhs_tag, rhs_tag, location))?,
        BoolOp::And => entry.append_op_result(arith::andi(lhs_tag, rhs_tag, location))?,
//        BoolOp::Xor => entry.append_op_result(arith::xori(lhs_tag, rhs_tag, location))?,
        BoolOp::Xor => entry.append_op_result(arith::xori(lhs_tag, rhs_tag, location))?,
//        BoolOp::Or => entry.append_op_result(arith::ori(lhs_tag, rhs_tag, location))?,
        BoolOp::Or => entry.append_op_result(arith::ori(lhs_tag, rhs_tag, location))?,
//    };
    };
//

//    let res = entry.append_op_result(llvm::undef(
    let res = entry.append_op_result(llvm::undef(
//        enum_ty.build(
        enum_ty.build(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &info.param_signatures()[0].ty,
            &info.param_signatures()[0].ty,
//        )?,
        )?,
//        location,
        location,
//    ))?;
    ))?;
//

//    let res = entry.insert_value(context, location, res, new_tag_value, 0)?;
    let res = entry.insert_value(context, location, res, new_tag_value, 0)?;
//

//    entry.append_operation(helper.br(0, &[res], location));
    entry.append_operation(helper.br(0, &[res], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `bool_not_impl` libfunc.
/// Generate MLIR operations for the `bool_not_impl` libfunc.
//pub fn build_bool_not<'ctx, 'this>(
pub fn build_bool_not<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
//    let tag_bits = enum_ty
    let tag_bits = enum_ty
//        .variants()
        .variants()
//        .expect("bool is a enum and has variants")
        .expect("bool is a enum and has variants")
//        .len()
        .len()
//        .next_power_of_two()
        .next_power_of_two()
//        .trailing_zeros();
        .trailing_zeros();
//    let tag_ty = IntegerType::new(context, tag_bits).into();
    let tag_ty = IntegerType::new(context, tag_bits).into();
//

//    let value = entry.argument(0)?.into();
    let value = entry.argument(0)?.into();
//    let tag_value = entry.extract_value(context, location, value, tag_ty, 0)?;
    let tag_value = entry.extract_value(context, location, value, tag_ty, 0)?;
//

//    let const_1 = entry.const_int_from_type(context, location, 1, tag_ty)?;
    let const_1 = entry.const_int_from_type(context, location, 1, tag_ty)?;
//

//    let new_tag_value = entry.append_op_result(arith::xori(tag_value, const_1, location))?;
    let new_tag_value = entry.append_op_result(arith::xori(tag_value, const_1, location))?;
//

//    let res = entry.append_op_result(llvm::undef(
    let res = entry.append_op_result(llvm::undef(
//        enum_ty.build(
        enum_ty.build(
//            context,
            context,
//            helper,
            helper,
//            registry,
            registry,
//            metadata,
            metadata,
//            &info.param_signatures()[0].ty,
            &info.param_signatures()[0].ty,
//        )?,
        )?,
//        location,
        location,
//    ))?;
    ))?;
//    let res = entry.insert_value(context, location, res, new_tag_value, 0)?;
    let res = entry.insert_value(context, location, res, new_tag_value, 0)?;
//

//    entry.append_operation(helper.br(0, &[res], location));
    entry.append_operation(helper.br(0, &[res], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `unbox` libfunc.
/// Generate MLIR operations for the `unbox` libfunc.
//pub fn build_bool_to_felt252<'ctx, 'this>(
pub fn build_bool_to_felt252<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
//    let felt252_ty = registry.build_type(
    let felt252_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//

//    let tag_bits = enum_ty
    let tag_bits = enum_ty
//        .variants()
        .variants()
//        .expect("bool is a enum and has variants")
        .expect("bool is a enum and has variants")
//        .len()
        .len()
//        .next_power_of_two()
        .next_power_of_two()
//        .trailing_zeros();
        .trailing_zeros();
//    let tag_ty = IntegerType::new(context, tag_bits).into();
    let tag_ty = IntegerType::new(context, tag_bits).into();
//

//    let value = entry.argument(0)?.into();
    let value = entry.argument(0)?.into();
//    let tag_value = entry.extract_value(context, location, value, tag_ty, 0)?;
    let tag_value = entry.extract_value(context, location, value, tag_ty, 0)?;
//

//    let result = entry.append_op_result(arith::extui(tag_value, felt252_ty, location))?;
    let result = entry.append_op_result(arith::extui(tag_value, felt252_ty, location))?;
//

//    entry.append_operation(helper.br(0, &[result], location));
    entry.append_operation(helper.br(0, &[result], location));
//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::{
    use crate::{
//        utils::test::{jit_enum, jit_struct, load_cairo, run_program},
        utils::test::{jit_enum, jit_struct, load_cairo, run_program},
//        values::JitValue,
        values::JitValue,
//    };
    };
//

//    #[test]
    #[test]
//    fn run_not() {
    fn run_not() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test(a: bool) -> bool {
            fn run_test(a: bool) -> bool {
//                !a
                !a
//            }
            }
//        );
        );
//

//        let result = run_program(&program, "run_test", &[jit_enum!(0, jit_struct!())]).return_value;
        let result = run_program(&program, "run_test", &[jit_enum!(0, jit_struct!())]).return_value;
//        assert_eq!(result, jit_enum!(1, jit_struct!()));
        assert_eq!(result, jit_enum!(1, jit_struct!()));
//

//        let result = run_program(&program, "run_test", &[jit_enum!(1, jit_struct!())]).return_value;
        let result = run_program(&program, "run_test", &[jit_enum!(1, jit_struct!())]).return_value;
//        assert_eq!(result, jit_enum!(0, jit_struct!()));
        assert_eq!(result, jit_enum!(0, jit_struct!()));
//    }
    }
//

//    #[test]
    #[test]
//    fn run_and() {
    fn run_and() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test(a: bool, b: bool) -> bool {
            fn run_test(a: bool, b: bool) -> bool {
//                a && b
                a && b
//            }
            }
//        );
        );
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(1, jit_struct!()));
        assert_eq!(result, jit_enum!(1, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(0, jit_struct!()));
        assert_eq!(result, jit_enum!(0, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(0, jit_struct!()));
        assert_eq!(result, jit_enum!(0, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(0, jit_struct!()));
        assert_eq!(result, jit_enum!(0, jit_struct!()));
//    }
    }
//

//    #[test]
    #[test]
//    fn run_xor() {
    fn run_xor() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test(a: bool, b: bool) -> bool {
            fn run_test(a: bool, b: bool) -> bool {
//                a ^ b
                a ^ b
//            }
            }
//        );
        );
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(0, jit_struct!()));
        assert_eq!(result, jit_enum!(0, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(1, jit_struct!()));
        assert_eq!(result, jit_enum!(1, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(1, jit_struct!()));
        assert_eq!(result, jit_enum!(1, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(0, jit_struct!()));
        assert_eq!(result, jit_enum!(0, jit_struct!()));
//    }
    }
//

//    #[test]
    #[test]
//    fn run_or() {
    fn run_or() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test(a: bool, b: bool) -> bool {
            fn run_test(a: bool, b: bool) -> bool {
//                a || b
                a || b
//            }
            }
//        );
        );
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(1, jit_struct!()));
        assert_eq!(result, jit_enum!(1, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(1, jit_struct!()));
        assert_eq!(result, jit_enum!(1, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(1, jit_struct!()));
        assert_eq!(result, jit_enum!(1, jit_struct!()));
//

//        let result = run_program(
        let result = run_program(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
//        )
        )
//        .return_value;
        .return_value;
//        assert_eq!(result, jit_enum!(0, jit_struct!()));
        assert_eq!(result, jit_enum!(0, jit_struct!()));
//    }
    }
//

//    #[test]
    #[test]
//    fn bool_to_felt252() {
    fn bool_to_felt252() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            fn run_test(a: bool) -> felt252 {
            fn run_test(a: bool) -> felt252 {
//                bool_to_felt252(a)
                bool_to_felt252(a)
//            }
            }
//        );
        );
//

//        let result = run_program(&program, "run_test", &[jit_enum!(1, jit_struct!())]).return_value;
        let result = run_program(&program, "run_test", &[jit_enum!(1, jit_struct!())]).return_value;
//        assert_eq!(result, JitValue::Felt252(1.into()));
        assert_eq!(result, JitValue::Felt252(1.into()));
//

//        let result = run_program(&program, "run_test", &[jit_enum!(0, jit_struct!())]).return_value;
        let result = run_program(&program, "run_test", &[jit_enum!(0, jit_struct!())]).return_value;
//        assert_eq!(result, JitValue::Felt252(0.into()));
        assert_eq!(result, JitValue::Felt252(0.into()));
//    }
    }
//}
}
