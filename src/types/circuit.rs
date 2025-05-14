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

/// Builds the circuit accumulator type.
///
/// ## Layout:
///
/// Holds up to N_INPUTS - 1 elements. Where each element is an u384.
///
/// ```rust
/// struct {
///     size: u64,
///     data: *u384,
/// }
/// ```
pub fn build_circuit_accumulator<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    _info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    let fields = vec![
        IntegerType::new(context, 64).into(),
        llvm::r#type::pointer(context, 0),
    ];

    Ok(llvm::r#type::r#struct(context, &fields, false))
}

/// Builds the circuit data type.
///
/// ## Layout:
///
/// Holds N_INPUTS elements. Where each element is an u384.
///
/// ```rust
/// struct {
///     data: *u384,
/// }
/// ```
pub fn build_circuit_data<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    _info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    Ok(llvm::r#type::pointer(context, 0))
}

/// Builds the circuit outputs type.
///
/// ## Layout:
///
/// Holds N_VALUES elements, where each element is a struct-shaped u384,
/// A struct-shaped u384 contains 4 limbs, each a u96.
///
/// ```rust
/// struct {
///     data: *u384_struct,
///     modulus: u384_struct,
/// };
/// ```
pub fn build_circuit_outputs<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    _info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    Ok(llvm::r#type::r#struct(
        context,
        &[
            llvm::r#type::pointer(context, 0),
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
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
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

        CircuitTypeConcrete::CircuitData(_) => Ok(Layout::new::<*mut ()>()),
        CircuitTypeConcrete::CircuitOutputs(_) => {
            let u384_struct_layout = layout_repeat(&get_integer_layout(96), 4)?.0;
            let pointer_layout = Layout::new::<*mut ()>();

            Ok(pointer_layout.extend(u384_struct_layout)?.0)
        }
        CircuitTypeConcrete::CircuitInputAccumulator(_) => {
            let integer_layout = get_integer_layout(64);
            let pointer_layout = Layout::new::<*mut ()>();

            Ok(integer_layout.extend(pointer_layout)?.0)
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
