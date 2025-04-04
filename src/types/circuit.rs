//! # `Circuit` type

use std::alloc::Layout;

use super::WithSelf;
use crate::{
    error::{Result, SierraAssertError},
    metadata::MetadataStorage,
    utils::{get_integer_layout, layout_repeat},
};
use cairo_lang_sierra::{
    extensions::{
        circuit::{CircuitTypeConcrete, ConcreteU96LimbsLessThanGuarantee},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        types::InfoOnlyConcreteType,
    },
    program::GenericArg,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    selector: WithSelf<CircuitTypeConcrete>,
) -> Result<Type<'ctx>> {
    match &*selector {
        CircuitTypeConcrete::CircuitModulus(_) => Ok(IntegerType::new(context, 384).into()),
        CircuitTypeConcrete::U96Guarantee(_) => Ok(IntegerType::new(context, 96).into()),
        CircuitTypeConcrete::CircuitInputAccumulator(info) => build_circuit_accumulator(
            context,
            module,
            registry,
            metadata,
            WithSelf::new(selector.self_ty(), info),
        ),
        CircuitTypeConcrete::CircuitData(info) => build_circuit_data(
            context,
            module,
            registry,
            metadata,
            WithSelf::new(selector.self_ty(), info),
        ),
        CircuitTypeConcrete::CircuitOutputs(info) => build_circuit_outputs(
            context,
            module,
            registry,
            metadata,
            WithSelf::new(selector.self_ty(), info),
        ),
        CircuitTypeConcrete::U96LimbsLessThanGuarantee(info) => {
            build_u96_limbs_less_than_guarantee(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(selector.self_ty(), info),
            )
        }
        // builtins
        CircuitTypeConcrete::AddMod(_) | CircuitTypeConcrete::MulMod(_) => {
            Ok(IntegerType::new(context, 64).into())
        }
        // noops
        CircuitTypeConcrete::CircuitDescriptor(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_)
        | CircuitTypeConcrete::CircuitPartialOutputs(_) => {
            Ok(llvm::r#type::array(IntegerType::new(context, 8).into(), 0))
        }
        // phantoms
        CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::AddModGate(_)
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::CircuitInput(_) => {
            Err(SierraAssertError::BadTypeInit(selector.self_ty.clone()))?
        }
    }
}

pub fn build_circuit_accumulator<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    let Some(GenericArg::Type(circuit_type_id)) = info.info.long_id.generic_args.first() else {
        return Err(SierraAssertError::BadTypeInfo.into());
    };
    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
        registry.get_type(circuit_type_id)?
    else {
        return Err(SierraAssertError::BadTypeInfo.into());
    };

    let n_inputs = circuit.circuit_info.n_inputs;

    let fields = vec![
        IntegerType::new(context, 64).into(),
        llvm::r#type::array(IntegerType::new(context, 384).into(), n_inputs as u32 - 1),
    ];

    Ok(llvm::r#type::r#struct(context, &fields, false))
}

pub fn build_circuit_data<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    let Some(GenericArg::Type(circuit_type_id)) = info.info.long_id.generic_args.first() else {
        return Err(SierraAssertError::BadTypeInfo.into());
    };
    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
        registry.get_type(circuit_type_id)?
    else {
        return Err(SierraAssertError::BadTypeInfo.into());
    };

    let n_inputs = circuit.circuit_info.n_inputs;

    Ok(llvm::r#type::array(
        IntegerType::new(context, 384).into(),
        n_inputs as u32,
    ))
}

pub fn build_circuit_outputs<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    let Some(GenericArg::Type(circuit_type_id)) = info.info.long_id.generic_args.first() else {
        return Err(SierraAssertError::BadTypeInfo.into());
    };
    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
        registry.get_type(circuit_type_id)?
    else {
        return Err(SierraAssertError::BadTypeInfo.into());
    };

    let n_gates = circuit.circuit_info.values.len();

    Ok(llvm::r#type::r#struct(
        context,
        &[
            llvm::r#type::array(build_u384_struct_type(context), n_gates as u32),
            build_u384_struct_type(context),
        ],
        false,
    ))
}

pub fn build_u96_limbs_less_than_guarantee<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: WithSelf<ConcreteU96LimbsLessThanGuarantee>,
) -> Result<Type<'ctx>> {
    let limbs = info.inner.limb_count;

    let u96_type = IntegerType::new(context, 96).into();
    let limb_struct_type = llvm::r#type::r#struct(context, &vec![u96_type; limbs], false);

    Ok(llvm::r#type::r#struct(
        context,
        &[limb_struct_type, limb_struct_type],
        false,
    ))
}

pub const fn is_complex(info: &CircuitTypeConcrete) -> bool {
    match *info {
        CircuitTypeConcrete::AddMod(_)
        | CircuitTypeConcrete::MulMod(_)
        | CircuitTypeConcrete::AddModGate(_)
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::U96Guarantee(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::U96LimbsLessThanGuarantee(_)
        | CircuitTypeConcrete::CircuitModulus(_)
        | CircuitTypeConcrete::CircuitInput(_)
        | CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::CircuitDescriptor(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_) => false,

        CircuitTypeConcrete::CircuitInputAccumulator(_)
        | CircuitTypeConcrete::CircuitPartialOutputs(_)
        | CircuitTypeConcrete::CircuitData(_)
        | CircuitTypeConcrete::CircuitOutputs(_) => true,
    }
}

pub const fn is_zst(info: &CircuitTypeConcrete) -> bool {
    match *info {
        CircuitTypeConcrete::AddModGate(_)
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::CircuitInput(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::U96LimbsLessThanGuarantee(_)
        | CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::CircuitDescriptor(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_) => true,

        CircuitTypeConcrete::AddMod(_)
        | CircuitTypeConcrete::CircuitModulus(_)
        | CircuitTypeConcrete::U96Guarantee(_)
        | CircuitTypeConcrete::MulMod(_)
        | CircuitTypeConcrete::CircuitInputAccumulator(_)
        | CircuitTypeConcrete::CircuitPartialOutputs(_)
        | CircuitTypeConcrete::CircuitData(_)
        | CircuitTypeConcrete::CircuitOutputs(_) => false,
    }
}

pub fn layout(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &CircuitTypeConcrete,
) -> Result<Layout> {
    match info {
        CircuitTypeConcrete::AddMod(_) | CircuitTypeConcrete::MulMod(_) => {
            Ok(get_integer_layout(64))
        }
        CircuitTypeConcrete::CircuitModulus(_) => Ok(get_integer_layout(384)),
        CircuitTypeConcrete::U96Guarantee(_) => Ok(get_integer_layout(96)),

        CircuitTypeConcrete::AddModGate(_)
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::CircuitInput(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::U96LimbsLessThanGuarantee(_)
        | CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::CircuitDescriptor(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_) => Ok(Layout::new::<()>()),

        CircuitTypeConcrete::CircuitData(info) => {
            let Some(GenericArg::Type(circuit_type_id)) = info.info.long_id.generic_args.first()
            else {
                return Err(SierraAssertError::BadTypeInfo.into());
            };
            let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
                registry.get_type(circuit_type_id)?
            else {
                return Err(SierraAssertError::BadTypeInfo.into());
            };

            let n_inputs = circuit.circuit_info.n_inputs;

            let u384_layout = get_integer_layout(384);

            let layout = layout_repeat(&u384_layout, n_inputs)?.0;

            Ok(layout)
        }
        CircuitTypeConcrete::CircuitOutputs(info) => {
            let Some(GenericArg::Type(circuit_type_id)) = info.info.long_id.generic_args.first()
            else {
                return Err(SierraAssertError::BadTypeInfo.into());
            };
            let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
                registry.get_type(circuit_type_id)?
            else {
                return Err(SierraAssertError::BadTypeInfo.into());
            };

            let n_gates = circuit.circuit_info.values.len();

            let u384_layout = get_integer_layout(384);

            let layout = layout_repeat(&u384_layout, n_gates)?.0;

            Ok(layout)
        }
        CircuitTypeConcrete::CircuitInputAccumulator(info) => {
            let Some(GenericArg::Type(circuit_type_id)) = info.info.long_id.generic_args.first()
            else {
                return Err(SierraAssertError::BadTypeInfo.into());
            };
            let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
                registry.get_type(circuit_type_id)?
            else {
                return Err(SierraAssertError::BadTypeInfo.into());
            };

            let n_inputs = circuit.circuit_info.n_inputs;

            let length_layout = get_integer_layout(64);

            let u384_layout = get_integer_layout(384);
            let inputs_layout = layout_repeat(&u384_layout, n_inputs - 1)?.0;
            let layout = length_layout.extend(inputs_layout)?.0;

            Ok(layout)
        }
        CircuitTypeConcrete::CircuitPartialOutputs(_) => Ok(Layout::new::<()>()),
    }
}

pub fn build_u384_struct_type(context: &Context) -> Type<'_> {
    llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
        ],
        false,
    )
}
