#![cfg(feature = "with-trace-dump")]

use std::collections::HashMap;

use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::{ConcreteTypeId, VarId},
    program::StatementIdx,
    program_registry::ProgramRegistry,
};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use melior::{
    ir::{BlockRef, Location, Module, Value, ValueLike},
    Context,
};

use crate::{
    metadata::{trace_dump::TraceDumpMeta, MetadataStorage},
    types::TypeBuilder,
};

use super::BlockExt;

#[allow(clippy::too_many_arguments)]
pub fn build_state_snapshot(
    context: &Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    module: &Module,
    block: &BlockRef,
    location: Location,
    metadata: &mut MetadataStorage,
    statement_idx: StatementIdx,
    state: &OrderedHashMap<VarId, Value>,
    var_types: &HashMap<VarId, ConcreteTypeId>,
) {
    let trace_dump = metadata.get_or_insert_with(TraceDumpMeta::default);

    for (var_id, value) in state.iter() {
        let value_type_id = var_types.get(var_id).unwrap();
        let value_type = registry.get_type(value_type_id).unwrap();

        let layout = value_type.layout(registry).unwrap();

        let ptr_value = block
            .alloca1(context, location, value.r#type(), layout.align())
            .unwrap();
        block.store(context, location, ptr_value, *value).unwrap();

        trace_dump
            .build_state(
                context,
                module,
                block,
                var_id,
                value_type_id,
                ptr_value,
                location,
            )
            .unwrap();
    }

    trace_dump
        .build_push(context, module, block, statement_idx, location)
        .unwrap();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cairo_lang_sierra::{program::Program, program_registry::ProgramRegistry};
    use pretty_assertions_sorted::assert_eq_sorted;
    use rstest::{fixture, rstest};
    use sierra_emu::{starknet::StubSyscallHandler, VirtualMachine};

    use crate::{
        context::NativeContext,
        executor::AotNativeExecutor,
        metadata::trace_dump::{
            trace_dump_runtime::{TraceDump, TRACE_DUMP},
            TraceBinding,
        },
        types::TypeBuilder,
        utils::test::load_cairo,
        OptLevel,
    };

    #[fixture]
    fn program() -> Program {
        let (_, program) = load_cairo! {
            use core::felt252;

            fn main() -> felt252 {
                let n = 10;
                let result = fib(1, 1, n);
                result
            }

            fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
                match n {
                    0 => a,
                    _ => fib(b, a + b, n - 1),
                }
            }
        };
        program
    }

    #[rstest]
    fn test_program(program: Program) {
        let entrypoint_function = &program
            .funcs
            .iter()
            .find(|x| {
                x.id.debug_name
                    .as_ref()
                    .map(|x| x.contains("main"))
                    .unwrap_or_default()
            })
            .unwrap()
            .clone();

        let native_context = NativeContext::new();
        let module = native_context
            .compile(&program, false, Some(Default::default()))
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default()).unwrap();

        if let Some(trace_id) = executor.find_symbol_ptr(TraceBinding::TraceId.symbol()) {
            let trace_id = trace_id.cast::<u64>();
            unsafe { *trace_id = 0 };
        }

        TRACE_DUMP.lock().unwrap().insert(
            0,
            TraceDump::new(ProgramRegistry::new(&program).unwrap(), |ty, registry| {
                ty.layout(registry).unwrap()
            }),
        );

        executor
            .invoke_dynamic(&entrypoint_function.id, &[], Some(u64::MAX))
            .unwrap();

        let native_trace = TRACE_DUMP
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .trace
            .clone();

        let mut vm = VirtualMachine::new(Arc::new(program));

        let initial_gas = u64::MAX;
        let args = [];
        vm.call_program(entrypoint_function, initial_gas, args.into_iter());

        let syscall_handler = &mut StubSyscallHandler::default();
        let emu_trace = vm.run_with_trace(syscall_handler);

        assert_eq_sorted!(emu_trace, native_trace);
    }
}
