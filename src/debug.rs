use crate::{
    error::Error,
    ffi::{
        mlirLLVMDIBasicTypeAttrGet, mlirLLVMDICompositeTypeAttrGet, mlirLLVMDIDerivedTypeAttrGet,
        mlirLLVMDIFileAttrGet, mlirLLVMDINullTypeAttrGet, MlirLLVMDWTag, MlirLLVMTypeEncoding,
    },
    types::TypeBuilder,
    utils::get_integer_layout,
};
use cairo_lang_sierra::{
    extensions::{
        array::ArrayConcreteLibfunc,
        boolean::BoolConcreteLibfunc,
        boxing::BoxConcreteLibfunc,
        bytes31::Bytes31ConcreteLibfunc,
        casts::CastConcreteLibfunc,
        const_type::ConstConcreteLibfunc,
        core::{CoreConcreteLibfunc, CoreLibfunc, CoreType, CoreTypeConcrete},
        coupon::CouponConcreteLibfunc,
        debug::DebugConcreteLibfunc,
        ec::EcConcreteLibfunc,
        enm::EnumConcreteLibfunc,
        felt252::{Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete},
        felt252_dict::{Felt252DictConcreteLibfunc, Felt252DictEntryConcreteLibfunc},
        gas::GasConcreteLibfunc,
        int::{
            signed::SintConcrete, signed128::Sint128Concrete, unsigned::UintConcrete,
            unsigned128::Uint128Concrete, unsigned256::Uint256Concrete,
            unsigned512::Uint512Concrete, IntOperator,
        },
        mem::MemConcreteLibfunc,
        nullable::NullableConcreteLibfunc,
        pedersen::PedersenConcreteLibfunc,
        poseidon::PoseidonConcreteLibfunc,
        starknet::{
            secp256::{Secp256ConcreteLibfunc, Secp256OpConcreteLibfunc},
            testing::TestingConcreteLibfunc,
            StarkNetConcreteLibfunc,
        },
        structure::StructConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{attribute::StringAttribute, Attribute, AttributeLike},
    Context,
};
use mlir_sys::MlirAttribute;
use std::alloc::Layout;

pub fn libfunc_to_name(value: &CoreConcreteLibfunc) -> &'static str {
    match value {
        CoreConcreteLibfunc::ApTracking(value) => match value {
            cairo_lang_sierra::extensions::ap_tracking::ApTrackingConcreteLibfunc::Revoke(_) => {
                "revoke_ap_tracking"
            }
            cairo_lang_sierra::extensions::ap_tracking::ApTrackingConcreteLibfunc::Enable(_) => {
                "enable_ap_tracking"
            }
            cairo_lang_sierra::extensions::ap_tracking::ApTrackingConcreteLibfunc::Disable(_) => {
                "disable_ap_tracking"
            }
        },
        CoreConcreteLibfunc::Array(value) => match value {
            ArrayConcreteLibfunc::New(_) => "array_new",
            ArrayConcreteLibfunc::SpanFromTuple(_) => "span_from_tuple",
            ArrayConcreteLibfunc::Append(_) => "array_append",
            ArrayConcreteLibfunc::PopFront(_) => "array_pop_front",
            ArrayConcreteLibfunc::PopFrontConsume(_) => "array_pop_front_consume",
            ArrayConcreteLibfunc::Get(_) => "array_get",
            ArrayConcreteLibfunc::Slice(_) => "array_slice",
            ArrayConcreteLibfunc::Len(_) => "array_len",
            ArrayConcreteLibfunc::SnapshotPopFront(_) => "array_snapshot_pop_front",
            ArrayConcreteLibfunc::SnapshotPopBack(_) => "array_snapshot_pop_back",
        },
        CoreConcreteLibfunc::BranchAlign(_) => "branch_align",
        CoreConcreteLibfunc::Bool(value) => match value {
            BoolConcreteLibfunc::And(_) => "bool_and",
            BoolConcreteLibfunc::Not(_) => "bool_not",
            BoolConcreteLibfunc::Xor(_) => "bool_xor",
            BoolConcreteLibfunc::Or(_) => "bool_or",
            BoolConcreteLibfunc::ToFelt252(_) => "bool_to_felt252",
        },
        CoreConcreteLibfunc::Box(value) => match value {
            BoxConcreteLibfunc::Into(_) => "box_into",
            BoxConcreteLibfunc::Unbox(_) => "box_unbox",
            BoxConcreteLibfunc::ForwardSnapshot(_) => "box_forward_snapshot",
        },
        CoreConcreteLibfunc::Cast(value) => match value {
            CastConcreteLibfunc::Downcast(_) => "downcast",
            CastConcreteLibfunc::Upcast(_) => "upcast",
        },
        CoreConcreteLibfunc::Coupon(value) => match value {
            CouponConcreteLibfunc::Buy(_) => "coupon_buy",
            CouponConcreteLibfunc::Refund(_) => "coupon_refund",
        },
        CoreConcreteLibfunc::CouponCall(_) => "coupon_call",
        CoreConcreteLibfunc::Drop(_) => "drop",
        CoreConcreteLibfunc::Dup(_) => "dup",
        CoreConcreteLibfunc::Ec(value) => match value {
            EcConcreteLibfunc::IsZero(_) => "ec_is_zero",
            EcConcreteLibfunc::Neg(_) => "ec_neg",
            EcConcreteLibfunc::StateAdd(_) => "ec_state_add",
            EcConcreteLibfunc::TryNew(_) => "ec_try_new",
            EcConcreteLibfunc::StateFinalize(_) => "ec_state_finalize",
            EcConcreteLibfunc::StateInit(_) => "ec_state_init",
            EcConcreteLibfunc::StateAddMul(_) => "ec_state_add_mul",
            EcConcreteLibfunc::PointFromX(_) => "ec_point_from_x",
            EcConcreteLibfunc::UnwrapPoint(_) => "ec_unwrap_point",
            EcConcreteLibfunc::Zero(_) => "ec_zero",
        },
        CoreConcreteLibfunc::Felt252(value) => match value {
            Felt252Concrete::BinaryOperation(op) => match op {
                Felt252BinaryOperationConcrete::WithVar(op) => match &op.operator {
                    Felt252BinaryOperator::Add => "felt252_add",
                    Felt252BinaryOperator::Sub => "felt252_sub",
                    Felt252BinaryOperator::Mul => "felt252_mul",
                    Felt252BinaryOperator::Div => "felt252_div",
                },
                Felt252BinaryOperationConcrete::WithConst(op) => match &op.operator {
                    Felt252BinaryOperator::Add => "felt252_const_add",
                    Felt252BinaryOperator::Sub => "felt252_const_sub",
                    Felt252BinaryOperator::Mul => "felt252_const_mul",
                    Felt252BinaryOperator::Div => "felt252_const_div",
                },
            },
            Felt252Concrete::Const(_) => "felt252_const",
            Felt252Concrete::IsZero(_) => "felt252_is_zero",
        },
        CoreConcreteLibfunc::Const(value) => match value {
            ConstConcreteLibfunc::AsBox(_) => "const_as_box",
            ConstConcreteLibfunc::AsImmediate(_) => "const_as_immediate",
        },
        CoreConcreteLibfunc::FunctionCall(_) => "function_call",
        CoreConcreteLibfunc::Gas(value) => match value {
            GasConcreteLibfunc::WithdrawGas(_) => "withdraw_gas",
            GasConcreteLibfunc::RedepositGas(_) => "redeposit_gas",
            GasConcreteLibfunc::GetAvailableGas(_) => "get_available_gas",
            GasConcreteLibfunc::BuiltinWithdrawGas(_) => "builtin_withdraw_gas",
            GasConcreteLibfunc::GetBuiltinCosts(_) => "get_builtin_costs",
        },
        CoreConcreteLibfunc::Uint8(value) => match value {
            UintConcrete::Const(_) => "u8_const",
            UintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u8_overflowing_add",
                IntOperator::OverflowingSub => "u8_overflowing_sub",
            },
            UintConcrete::SquareRoot(_) => "u8_sqrt",
            UintConcrete::Equal(_) => "u8_eq",
            UintConcrete::ToFelt252(_) => "u8_to_felt252",
            UintConcrete::FromFelt252(_) => "u8_from_felt252",
            UintConcrete::IsZero(_) => "u8_is_zero",
            UintConcrete::Divmod(_) => "u8_divmod",
            UintConcrete::WideMul(_) => "u8_wide_mul",
            UintConcrete::Bitwise(_) => "u8_bitwise",
        },
        CoreConcreteLibfunc::Uint16(value) => match value {
            UintConcrete::Const(_) => "u16_const",
            UintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u16_overflowing_add",
                IntOperator::OverflowingSub => "u16_overflowing_sub",
            },
            UintConcrete::SquareRoot(_) => "u16_sqrt",
            UintConcrete::Equal(_) => "u16_eq",
            UintConcrete::ToFelt252(_) => "u16_to_felt252",
            UintConcrete::FromFelt252(_) => "u16_from_felt252",
            UintConcrete::IsZero(_) => "u16_is_zero",
            UintConcrete::Divmod(_) => "u16_divmod",
            UintConcrete::WideMul(_) => "u16_wide_mul",
            UintConcrete::Bitwise(_) => "u16_bitwise",
        },
        CoreConcreteLibfunc::Uint32(value) => match value {
            UintConcrete::Const(_) => "u32_const",
            UintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u32_overflowing_add",
                IntOperator::OverflowingSub => "u32_overflowing_sub",
            },
            UintConcrete::SquareRoot(_) => "u32_sqrt",
            UintConcrete::Equal(_) => "u32_eq",
            UintConcrete::ToFelt252(_) => "u32_to_felt252",
            UintConcrete::FromFelt252(_) => "u32_from_felt252",
            UintConcrete::IsZero(_) => "u32_is_zero",
            UintConcrete::Divmod(_) => "u32_divmod",
            UintConcrete::WideMul(_) => "u32_wide_mul",
            UintConcrete::Bitwise(_) => "u32_bitwise",
        },
        CoreConcreteLibfunc::Uint64(value) => match value {
            UintConcrete::Const(_) => "u64_const",
            UintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u64_overflowing_add",
                IntOperator::OverflowingSub => "u64_overflowing_sub",
            },
            UintConcrete::SquareRoot(_) => "u64_sqrt",
            UintConcrete::Equal(_) => "u64_eq",
            UintConcrete::ToFelt252(_) => "u64_to_felt252",
            UintConcrete::FromFelt252(_) => "u64_from_felt252",
            UintConcrete::IsZero(_) => "u64_is_zero",
            UintConcrete::Divmod(_) => "u64_divmod",
            UintConcrete::WideMul(_) => "u64_wide_mul",
            UintConcrete::Bitwise(_) => "u64_bitwise",
        },
        CoreConcreteLibfunc::Uint128(value) => match value {
            Uint128Concrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "u128_overflowing_add",
                IntOperator::OverflowingSub => "u128_overflowing_sub",
            },
            Uint128Concrete::Divmod(_) => "u128_divmod",
            Uint128Concrete::GuaranteeMul(_) => "u128_guarantee_mul",
            Uint128Concrete::MulGuaranteeVerify(_) => "u128_mul_guarantee_verify",
            Uint128Concrete::Equal(_) => "u128_equal",
            Uint128Concrete::SquareRoot(_) => "u128_sqrt",
            Uint128Concrete::Const(_) => "u128_const",
            Uint128Concrete::FromFelt252(_) => "u128_from_felt",
            Uint128Concrete::ToFelt252(_) => "u128_to_felt252",
            Uint128Concrete::IsZero(_) => "u128_is_zero",
            Uint128Concrete::Bitwise(_) => "u128_bitwise",
            Uint128Concrete::ByteReverse(_) => "u128_bytereverse",
        },
        CoreConcreteLibfunc::Uint256(value) => match value {
            Uint256Concrete::IsZero(_) => "u256_is_zero",
            Uint256Concrete::Divmod(_) => "u256_divmod",
            Uint256Concrete::SquareRoot(_) => "u256_sqrt",
            Uint256Concrete::InvModN(_) => "u256_inv_mod_n",
        },
        CoreConcreteLibfunc::Uint512(value) => match value {
            Uint512Concrete::DivModU256(_) => "u512_divmod_u256",
        },
        CoreConcreteLibfunc::Sint8(value) => match value {
            SintConcrete::Const(_) => "i8_const",
            SintConcrete::Equal(_) => "i8_eq",
            SintConcrete::ToFelt252(_) => "i8_to_felt252",
            SintConcrete::FromFelt252(_) => "i8_from_felt252",
            SintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i8_overflowing_add",
                IntOperator::OverflowingSub => "i8_overflowing_sub",
            },
            SintConcrete::Diff(_) => "i8_diff",
            SintConcrete::IsZero(_) => "i8_is_zero",
            SintConcrete::WideMul(_) => "i8_wide_mul",
        },
        CoreConcreteLibfunc::Sint16(value) => match value {
            SintConcrete::Const(_) => "i16_const",
            SintConcrete::Equal(_) => "i16_eq",
            SintConcrete::ToFelt252(_) => "i16_to_felt252",
            SintConcrete::FromFelt252(_) => "i16_from_felt252",
            SintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i16_overflowing_add",
                IntOperator::OverflowingSub => "i16_overflowing_sub",
            },
            SintConcrete::Diff(_) => "i16_diff",
            SintConcrete::IsZero(_) => "i16_is_zero",
            SintConcrete::WideMul(_) => "i16_wide_mul",
        },
        CoreConcreteLibfunc::Sint32(value) => match value {
            SintConcrete::Const(_) => "i32_const",
            SintConcrete::Equal(_) => "i32_eq",
            SintConcrete::ToFelt252(_) => "i32_to_felt252",
            SintConcrete::FromFelt252(_) => "i32_from_felt252",
            SintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i32_overflowing_add",
                IntOperator::OverflowingSub => "i32_overflowing_sub",
            },
            SintConcrete::Diff(_) => "i32_diff",
            SintConcrete::IsZero(_) => "i32_is_zero",
            SintConcrete::WideMul(_) => "i32_wide_mul",
        },
        CoreConcreteLibfunc::Sint64(value) => match value {
            SintConcrete::Const(_) => "i64_const",
            SintConcrete::Equal(_) => "i64_eq",
            SintConcrete::ToFelt252(_) => "i64_to_felt252",
            SintConcrete::FromFelt252(_) => "i64_from_felt252",
            SintConcrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i64_overflowing_add",
                IntOperator::OverflowingSub => "i64_overflowing_sub",
            },
            SintConcrete::Diff(_) => "i64_diff",
            SintConcrete::IsZero(_) => "i64_is_zero",
            SintConcrete::WideMul(_) => "i64_wide_mul",
        },
        CoreConcreteLibfunc::Sint128(value) => match value {
            Sint128Concrete::Const(_) => "i128_const",
            Sint128Concrete::Equal(_) => "i128_eq",
            Sint128Concrete::ToFelt252(_) => "i128_to_felt252",
            Sint128Concrete::FromFelt252(_) => "i128_from_felt252",
            Sint128Concrete::Operation(op) => match &op.operator {
                IntOperator::OverflowingAdd => "i128_overflowing_add",
                IntOperator::OverflowingSub => "i128_overflowing_sub",
            },
            Sint128Concrete::Diff(_) => "i128_diff",
            Sint128Concrete::IsZero(_) => "i128_is_zero",
        },
        CoreConcreteLibfunc::Mem(value) => match value {
            MemConcreteLibfunc::StoreTemp(_) => "store_temp",
            MemConcreteLibfunc::StoreLocal(_) => "store_local",
            MemConcreteLibfunc::FinalizeLocals(_) => "finalize_locals",
            MemConcreteLibfunc::AllocLocal(_) => "alloc_local",
            MemConcreteLibfunc::Rename(_) => "rename",
        },
        CoreConcreteLibfunc::Nullable(value) => match value {
            NullableConcreteLibfunc::Null(_) => "nullable_null",
            NullableConcreteLibfunc::NullableFromBox(_) => "nullable_from_box",
            NullableConcreteLibfunc::MatchNullable(_) => "match_nullable",
            NullableConcreteLibfunc::ForwardSnapshot(_) => "nullable_forward_snapshot",
        },
        CoreConcreteLibfunc::UnwrapNonZero(_) => "unwrap_non_zero",
        CoreConcreteLibfunc::UnconditionalJump(_) => "jump",
        CoreConcreteLibfunc::Enum(value) => match value {
            EnumConcreteLibfunc::Init(_) => "enum_init",
            EnumConcreteLibfunc::FromBoundedInt(_) => "enum_from_bounded_int",
            EnumConcreteLibfunc::Match(_) => "enum_match",
            EnumConcreteLibfunc::SnapshotMatch(_) => "enum_snapshot_match",
        },
        CoreConcreteLibfunc::Struct(value) => match value {
            StructConcreteLibfunc::Construct(_) => "struct_construct",
            StructConcreteLibfunc::Deconstruct(_) => "struct_deconstruct",
            StructConcreteLibfunc::SnapshotDeconstruct(_) => "struct_snapshot_deconstruct",
        },
        CoreConcreteLibfunc::Felt252Dict(value) => match value {
            Felt252DictConcreteLibfunc::New(_) => "felt252dict_new",
            Felt252DictConcreteLibfunc::Squash(_) => "felt252dict_squash",
        },
        CoreConcreteLibfunc::Felt252DictEntry(value) => match value {
            Felt252DictEntryConcreteLibfunc::Get(_) => "felt252dict_get",
            Felt252DictEntryConcreteLibfunc::Finalize(_) => "felt252dict_finalize",
        },
        CoreConcreteLibfunc::Pedersen(value) => match value {
            PedersenConcreteLibfunc::PedersenHash(_) => "pedersen_hash",
        },
        CoreConcreteLibfunc::Poseidon(value) => match value {
            PoseidonConcreteLibfunc::HadesPermutation(_) => "hades_permutation",
        },
        CoreConcreteLibfunc::StarkNet(value) => match value {
            StarkNetConcreteLibfunc::CallContract(_) => "call_contract",
            StarkNetConcreteLibfunc::ClassHashConst(_) => "class_hash_const",
            StarkNetConcreteLibfunc::ClassHashTryFromFelt252(_) => "class_hash_try_from_felt252",
            StarkNetConcreteLibfunc::ClassHashToFelt252(_) => "class_hash_to_felt252",
            StarkNetConcreteLibfunc::ContractAddressConst(_) => "contract_address_const",
            StarkNetConcreteLibfunc::ContractAddressTryFromFelt252(_) => {
                "contract_address_try_from_felt252"
            }
            StarkNetConcreteLibfunc::ContractAddressToFelt252(_) => "contract_address_to_felt252",
            StarkNetConcreteLibfunc::StorageRead(_) => "storage_read",
            StarkNetConcreteLibfunc::StorageWrite(_) => "storage_write",
            StarkNetConcreteLibfunc::StorageBaseAddressConst(_) => "storage_base_address_const",
            StarkNetConcreteLibfunc::StorageBaseAddressFromFelt252(_) => {
                "storage_base_address_from_felt252"
            }
            StarkNetConcreteLibfunc::StorageAddressFromBase(_) => "storage_address_from_bas",
            StarkNetConcreteLibfunc::StorageAddressFromBaseAndOffset(_) => {
                "storage_address_from_Base_and_offset"
            }
            StarkNetConcreteLibfunc::StorageAddressToFelt252(_) => "storage_address_to_felt252",
            StarkNetConcreteLibfunc::StorageAddressTryFromFelt252(_) => {
                "storage_address_try_from_felt252"
            }
            StarkNetConcreteLibfunc::EmitEvent(_) => "emit_event",
            StarkNetConcreteLibfunc::GetBlockHash(_) => "get_block_hash",
            StarkNetConcreteLibfunc::GetExecutionInfo(_) => "get_exec_info_v1",
            StarkNetConcreteLibfunc::GetExecutionInfoV2(_) => "get_exec_info_v2",
            StarkNetConcreteLibfunc::Deploy(_) => "deploy",
            StarkNetConcreteLibfunc::Keccak(_) => "keccak",
            StarkNetConcreteLibfunc::LibraryCall(_) => "library_call",
            StarkNetConcreteLibfunc::ReplaceClass(_) => "replace_class",
            StarkNetConcreteLibfunc::SendMessageToL1(_) => "send_message_to_l1",
            StarkNetConcreteLibfunc::Testing(value) => match value {
                TestingConcreteLibfunc::Cheatcode(_) => "cheatcode",
            },
            StarkNetConcreteLibfunc::Secp256(value) => match value {
                Secp256ConcreteLibfunc::K1(value) => match value {
                    Secp256OpConcreteLibfunc::New(_) => "secp256k1_new",
                    Secp256OpConcreteLibfunc::Add(_) => "secp256k1_add",
                    Secp256OpConcreteLibfunc::Mul(_) => "secp256k1_mul",
                    Secp256OpConcreteLibfunc::GetPointFromX(_) => "secp256k1_get_point_from_x",
                    Secp256OpConcreteLibfunc::GetXy(_) => "secp256k1_get_xy",
                },
                Secp256ConcreteLibfunc::R1(value) => match value {
                    Secp256OpConcreteLibfunc::New(_) => "secp256r1_new",
                    Secp256OpConcreteLibfunc::Add(_) => "secp256r1_add",
                    Secp256OpConcreteLibfunc::Mul(_) => "secp256r1_mul",
                    Secp256OpConcreteLibfunc::GetPointFromX(_) => "secp256r1_get_point_from_x",
                    Secp256OpConcreteLibfunc::GetXy(_) => "secp256r1_get_xy",
                },
            },
        },
        CoreConcreteLibfunc::Debug(value) => match value {
            DebugConcreteLibfunc::Print(_) => "debug_print",
        },
        CoreConcreteLibfunc::SnapshotTake(_) => "snapshot_take",
        CoreConcreteLibfunc::Bytes31(value) => match value {
            Bytes31ConcreteLibfunc::Const(_) => "bytes31_const",
            Bytes31ConcreteLibfunc::ToFelt252(_) => "bytes31_to_felt252",
            Bytes31ConcreteLibfunc::TryFromFelt252(_) => "bytes31_try_from_felt252",
        },
    }
}

pub fn coretype_to_debugtype<'c>(
    context: &'c Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    ty: &CoreTypeConcrete,
    debug_name: Option<&str>,
) -> Result<Attribute<'c>, Error> {
    Ok(match ty {
        CoreTypeConcrete::Array(_) => todo!(),
        CoreTypeConcrete::Coupon(_) => todo!(),
        CoreTypeConcrete::Bitwise(_) => todo!(),
        CoreTypeConcrete::Box(inner) => {
            let inner_ty = registry.get_type(&inner.ty)?;
            let inner_debugty = coretype_to_debugtype(
                context,
                registry,
                inner_ty,
                inner.ty.debug_name.as_ref().map(|x| x.as_str()),
            )?;

            unsafe {
                Attribute::from_raw(mlirLLVMDIDerivedTypeAttrGet(
                    context.to_raw(),
                    MlirLLVMDWTag::DW_TAG_pointer_type,
                    StringAttribute::new(context, debug_name.unwrap_or("Box")).to_raw(),
                    inner_debugty.to_raw(),
                    64,
                    64,
                    0,
                ))
            }
        }
        CoreTypeConcrete::Const(inner) => {
            let inner_ty = registry.get_type(&inner.inner_ty)?;
            let layout_inner = inner_ty.layout(registry)?;
            let debug_attr = coretype_to_debugtype(
                context,
                registry,
                inner_ty,
                inner.inner_ty.debug_name.as_ref().map(|x| x.as_str()),
            )?;
            let derived_attr = create_derived_debug_type(
                context,
                debug_name.unwrap_or(&format!(
                    "Const<id:{}, {:?}>",
                    inner.inner_ty.id, inner.inner_data
                )),
                debug_attr,
                (layout_inner.size() * 8) as u64,
                (layout_inner.align() * 8) as u64,
                0,
                MlirLLVMDWTag::DW_TAG_base_type,
            );
            derived_attr
        }
        CoreTypeConcrete::EcOp(_) => todo!(),
        CoreTypeConcrete::EcPoint(_) => todo!(),
        CoreTypeConcrete::EcState(_) => {
            let mut types = Vec::new();
            let layout = ty.layout(registry)?;

            let mut acc_layout = Layout::new::<()>();

            let debug_attr = create_di_basic_type(
                context,
                "felt252",
                252,
                MlirLLVMTypeEncoding::Unsigned,
                MlirLLVMDWTag::DW_TAG_base_type,
            );

            let layout_inner = get_integer_layout(252);

            for i in 0..4 {
                let (cur_layout, offset) = acc_layout.extend(layout_inner).unwrap();
                acc_layout = cur_layout;

                let derived_attr = create_derived_debug_type(
                    context,
                    &format!("field_{}", i),
                    debug_attr,
                    (layout_inner.size() * 8) as u64,
                    (layout_inner.align() * 8) as u64,
                    (offset * 8) as u64,
                    MlirLLVMDWTag::DW_TAG_member,
                );
                types.push(derived_attr.to_raw());
            }

            create_composite_debug_type(
                context,
                debug_name.unwrap_or("Struct"),
                unsafe { Attribute::from_raw(mlirLLVMDINullTypeAttrGet(context.to_raw())) },
                (layout.size() * 8) as u64,
                (layout.align() * 8) as u64,
                MlirLLVMDWTag::DW_TAG_structure_type,
                &types,
            )
        }
        CoreTypeConcrete::Felt252(_) => create_di_basic_type(
            context,
            "felt252",
            252,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::GasBuiltin(_) => create_di_basic_type(
            context,
            "GasBuiltin",
            128,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::BuiltinCosts(_) => todo!(),
        CoreTypeConcrete::Uint8(_) => create_di_basic_type(
            context,
            "u8",
            8,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Uint16(_) => create_di_basic_type(
            context,
            "u16",
            16,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Uint32(_) => create_di_basic_type(
            context,
            "u32",
            32,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Uint64(_) => create_di_basic_type(
            context,
            "u64",
            64,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Uint128(_) => create_di_basic_type(
            context,
            "u128",
            128,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
        CoreTypeConcrete::Sint8(_) => create_di_basic_type(
            context,
            "s8",
            8,
            MlirLLVMTypeEncoding::Signed,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Sint16(_) => create_di_basic_type(
            context,
            "s16",
            16,
            MlirLLVMTypeEncoding::Signed,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Sint32(_) => create_di_basic_type(
            context,
            "s32",
            32,
            MlirLLVMTypeEncoding::Signed,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Sint64(_) => create_di_basic_type(
            context,
            "s64",
            64,
            MlirLLVMTypeEncoding::Signed,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Sint128(_) => create_di_basic_type(
            context,
            "s128",
            128,
            MlirLLVMTypeEncoding::Signed,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::NonZero(_) => todo!(),
        CoreTypeConcrete::Nullable(inner) => {
            let inner_ty = registry.get_type(&inner.ty)?;
            let inner_debugty = coretype_to_debugtype(
                context,
                registry,
                inner_ty,
                inner.ty.debug_name.as_ref().map(|x| x.as_str()),
            )?;

            unsafe {
                Attribute::from_raw(mlirLLVMDIDerivedTypeAttrGet(
                    context.to_raw(),
                    MlirLLVMDWTag::DW_TAG_pointer_type,
                    StringAttribute::new(context, debug_name.unwrap_or("Nullable")).to_raw(),
                    inner_debugty.to_raw(),
                    64,
                    64,
                    0,
                ))
            }
        }
        CoreTypeConcrete::RangeCheck(_) => create_di_basic_type(
            context,
            debug_name.unwrap_or("RangeCheck"),
            64,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Uninitialized(inner) => {
            let inner_ty = registry.get_type(&inner.ty)?;
            let layout_inner = inner_ty.layout(registry)?;
            let debug_attr = coretype_to_debugtype(
                context,
                registry,
                inner_ty,
                inner.ty.debug_name.as_ref().map(|x| x.as_str()),
            )?;
            let derived_attr = create_derived_debug_type(
                context,
                debug_name.unwrap_or(&format!("Uninitalized<id:{}>", inner.ty.id)),
                debug_attr,
                (layout_inner.size() * 8) as u64,
                (layout_inner.align() * 8) as u64,
                0,
                MlirLLVMDWTag::DW_TAG_base_type,
            );
            derived_attr
        }
        CoreTypeConcrete::Enum(_) => todo!(),
        CoreTypeConcrete::Struct(x) => {
            let mut types = Vec::new();
            let layout = ty.layout(registry)?;

            let mut acc_layout = Layout::new::<()>();

            for (i, field_ty_id) in x.members.iter().enumerate() {
                let field_ty = registry.get_type(field_ty_id)?;
                let layout_inner = field_ty.layout(registry)?;
                let (cur_layout, offset) = acc_layout.extend(layout_inner).unwrap();
                acc_layout = cur_layout;
                let debug_attr = coretype_to_debugtype(
                    context,
                    registry,
                    field_ty,
                    field_ty_id.debug_name.as_ref().map(|x| x.as_str()),
                )?;
                let derived_attr = create_derived_debug_type(
                    context,
                    &format!("field_{}", i),
                    debug_attr,
                    (layout_inner.size() * 8) as u64,
                    (layout_inner.align() * 8) as u64,
                    (offset * 8) as u64,
                    MlirLLVMDWTag::DW_TAG_member,
                );
                types.push(derived_attr.to_raw());
            }

            create_composite_debug_type(
                context,
                debug_name.unwrap_or("Struct"),
                unsafe { Attribute::from_raw(mlirLLVMDINullTypeAttrGet(context.to_raw())) },
                (layout.size() * 8) as u64,
                (layout.align() * 8) as u64,
                MlirLLVMDWTag::DW_TAG_structure_type,
                &types,
            )
        }
        CoreTypeConcrete::Felt252Dict(_) => todo!(),
        CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
        CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
        CoreTypeConcrete::Pedersen(_) => create_di_basic_type(
            context,
            debug_name.unwrap_or("Pedersen"),
            64,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Poseidon(_) => create_di_basic_type(
            context,
            debug_name.unwrap_or("Poseidon"),
            64,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::Span(_) => todo!(),
        CoreTypeConcrete::StarkNet(_) => todo!(),
        CoreTypeConcrete::SegmentArena(_) => todo!(),
        CoreTypeConcrete::Snapshot(_) => todo!(),
        CoreTypeConcrete::Bytes31(_) => create_di_basic_type(
            context,
            debug_name.unwrap_or("Bytes31"),
            248,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
        CoreTypeConcrete::BoundedInt(x) => create_di_basic_type(
            context,
            debug_name.unwrap_or(&format!("BoundedInt<{}, {}>", x.range.lower, x.range.upper)),
            252,
            MlirLLVMTypeEncoding::Unsigned,
            MlirLLVMDWTag::DW_TAG_base_type,
        ),
    })
}

/// Creates a LLVM debug basic type.
pub fn create_di_basic_type<'c>(
    context: &'c Context,
    name: &str,
    size_in_bits: u64,
    encoding: MlirLLVMTypeEncoding,
    tag: MlirLLVMDWTag,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(mlirLLVMDIBasicTypeAttrGet(
            context.to_raw(),
            tag as u32,
            StringAttribute::new(context, name).to_raw(),
            size_in_bits,
            encoding,
        ))
    }
}

pub fn create_derived_debug_type<'c>(
    context: &'c Context,
    name: &str,
    base_type: Attribute<'c>,
    size_in_bits: u64,
    align_in_bits: u64,
    offset_in_bits: u64,
    tag: MlirLLVMDWTag,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(mlirLLVMDIDerivedTypeAttrGet(
            context.to_raw(),
            tag,
            StringAttribute::new(context, name).to_raw(),
            base_type.to_raw(),
            size_in_bits,
            align_in_bits,
            offset_in_bits,
        ))
    }
}

pub fn create_composite_debug_type<'c>(
    context: &'c Context,
    name: &str,
    base_type: Attribute<'c>,
    size_in_bits: u64,
    align_in_bits: u64,
    tag: MlirLLVMDWTag,
    elements: &[MlirAttribute],
) -> Attribute<'c> {
    let file_attr = unsafe {
        mlirLLVMDIFileAttrGet(
            context.to_raw(),
            StringAttribute::new(context, "<unknown>").to_raw(),
            StringAttribute::new(context, "").to_raw(),
        )
    };

    unsafe {
        Attribute::from_raw(mlirLLVMDICompositeTypeAttrGet(
            context.to_raw(),
            tag,
            StringAttribute::new(context, name).to_raw(),
            file_attr,
            0,
            file_attr,
            base_type.to_raw(),
            3, // public
            size_in_bits,
            align_in_bits,
            elements.len(),
            elements.as_ptr(),
        ))
    }
}
