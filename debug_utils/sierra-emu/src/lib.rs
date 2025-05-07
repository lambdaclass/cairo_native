use std::{num::ParseIntError, str::FromStr, sync::Arc};

use cairo_lang_sierra::{
    extensions::{
        circuit::CircuitTypeConcrete,
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::StarknetTypeConcrete,
    },
    ids::ConcreteTypeId,
    program::{GenFunction, Program, StatementIdx},
    program_registry::ProgramRegistry,
};
use starknet::StubSyscallHandler;

pub use self::{dump::*, gas::BuiltinCosts, value::*, vm::VirtualMachine};

mod debug;
mod dump;
mod gas;
pub mod starknet;
mod test_utils;
mod value;
mod vm;

#[derive(Clone, Debug)]
pub enum EntryPoint {
    Number(u64),
    String(String),
}

impl FromStr for EntryPoint {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.chars().next() {
            Some(x) if x.is_numeric() => Self::Number(s.parse()?),
            _ => Self::String(s.to_string()),
        })
    }
}

pub fn find_entry_point_by_idx(
    program: &Program,
    entry_point_idx: usize,
) -> Option<&GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.id == entry_point_idx as u64)
}

pub fn find_entry_point_by_name<'a>(
    program: &'a Program,
    name: &str,
) -> Option<&'a GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_ref().map(|x| x.as_str()) == Some(name))
}

// If type is invisible to sierra (i.e. a single element container),
// finds it's actual concrete type recursively.
// If not, returns the current type
pub fn find_real_type(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    ty: &ConcreteTypeId,
) -> ConcreteTypeId {
    match registry.get_type(ty).unwrap() {
        cairo_lang_sierra::extensions::core::CoreTypeConcrete::Box(info) => {
            find_real_type(registry, &info.ty)
        }
        cairo_lang_sierra::extensions::core::CoreTypeConcrete::Uninitialized(info) => {
            find_real_type(registry, &info.ty)
        }
        cairo_lang_sierra::extensions::core::CoreTypeConcrete::Span(info) => {
            find_real_type(registry, &info.ty)
        }
        cairo_lang_sierra::extensions::core::CoreTypeConcrete::Snapshot(info) => {
            find_real_type(registry, &info.ty)
        }
        _ => ty.clone(),
    }
}

pub fn run_program(
    program: Arc<Program>,
    entry_point: EntryPoint,
    args: Vec<String>,
    available_gas: u64,
) -> ProgramTrace {
    let mut vm = VirtualMachine::new(program.clone());

    let function = program
        .funcs
        .iter()
        .find(|f| match &entry_point {
            EntryPoint::Number(x) => f.id.id == *x,
            EntryPoint::String(x) => f.id.debug_name.as_deref() == Some(x.as_str()),
        })
        .unwrap();

    let mut iter = args.into_iter();
    vm.push_frame(
        function.id.clone(),
        function
            .signature
            .param_types
            .iter()
            .map(|type_id| {
                let type_info = vm.registry().get_type(type_id).unwrap();
                match type_info {
                    CoreTypeConcrete::Felt252(_) => Value::parse_felt(&iter.next().unwrap()),
                    CoreTypeConcrete::GasBuiltin(_) => Value::U64(available_gas),
                    CoreTypeConcrete::RangeCheck(_)
                    | CoreTypeConcrete::RangeCheck96(_)
                    | CoreTypeConcrete::Bitwise(_)
                    | CoreTypeConcrete::Pedersen(_)
                    | CoreTypeConcrete::Poseidon(_)
                    | CoreTypeConcrete::SegmentArena(_)
                    | CoreTypeConcrete::Circuit(
                        CircuitTypeConcrete::AddMod(_) | CircuitTypeConcrete::MulMod(_),
                    ) => Value::Unit,
                    CoreTypeConcrete::Starknet(inner) => match inner {
                        StarknetTypeConcrete::System(_) => Value::Unit,
                        StarknetTypeConcrete::ClassHash(_)
                        | StarknetTypeConcrete::ContractAddress(_)
                        | StarknetTypeConcrete::StorageBaseAddress(_)
                        | StarknetTypeConcrete::StorageAddress(_) => {
                            Value::parse_felt(&iter.next().unwrap())
                        }
                        _ => todo!(),
                    },
                    _ => todo!(),
                }
            })
            .collect::<Vec<_>>(),
    );

    let syscall_handler = &mut StubSyscallHandler::default();

    vm.run_with_trace(syscall_handler)
}
