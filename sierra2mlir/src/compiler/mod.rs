use std::collections::{HashMap, HashSet};

use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        gas::{
            BuiltinCostWithdrawGasLibfunc, CostTokenType, GetAvailableGasLibfunc,
            RedepositGasLibfunc, WithdrawGasLibfunc,
        },
        NamedLibfunc,
    },
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_lang_sierra_to_casm::metadata::{calc_metadata, Metadata};
use color_eyre::eyre::bail;
use melior_next::{
    dialect,
    ir::{operation, Block, Location, Module, NamedAttribute, OperationRef, Region, Type},
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};
use tracing::debug;

use crate::{
    libfuncs::lib_func_def::SierraLibFunc, sierra_type::SierraType,
    userfuncs::user_func_def::UserFuncDef,
};

use self::fn_attributes::FnAttributes;

pub mod external_funcs;
pub mod fn_attributes;
pub mod helpers;
pub mod mlir_ops;
pub mod types;

pub struct CompilerGas {
    pub metadata: Metadata,
    /// Initial available gas.
    pub available_gas: usize,
}

pub struct Compiler<'ctx> {
    pub program: &'ctx Program,
    pub registry: ProgramRegistry<CoreType, CoreLibfunc>,
    pub gas: Option<CompilerGas>,
    pub context: Context,
    pub module: Module<'ctx>,
    pub main_print: bool,
    pub print_fd: i32,
}

/// Types, functions, etc storage.
/// This aproach works better with lifetimes.
#[derive(Debug, Default, Clone)]
pub struct Storage<'ctx> {
    pub(crate) types: HashMap<String, SierraType<'ctx>>,
    pub(crate) libfuncs: HashMap<String, SierraLibFunc<'ctx>>,
    pub(crate) userfuncs: HashMap<String, UserFuncDef<'ctx>>,
    pub(crate) helperfuncs: HashSet<String>,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(
        program: &'ctx Program,
        main_print: bool,
        print_fd: i32,
        available_gas: Option<usize>,
    ) -> color_eyre::Result<Self> {
        let sierra_program_registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;

        if available_gas.is_none()
            && program.type_declarations.iter().any(|decl| {
                matches!(
                    decl.long_id.generic_id.0.as_str(),
                    WithdrawGasLibfunc::STR_ID
                        | BuiltinCostWithdrawGasLibfunc::STR_ID
                        | RedepositGasLibfunc::STR_ID
                        | GetAvailableGasLibfunc::STR_ID
                )
            })
        {
            bail!("Program requires gas counter, please provide `--available-gas` argument.");
        }

        // Create the compiler gas structure if gas was provided.
        // source: https://github.com/starkware-libs/cairo/blob/v1.0.0-rc0/crates/cairo-lang-sierra-to-casm/src/metadata.rs
        let gas: Option<CompilerGas> = if let Some(mut available_gas) = available_gas {
            let metadata = calc_metadata(program, Default::default())?;

            debug!("available gas before entry point = {}", available_gas);

            /*
               spend the constant known needed gas
               here we assume main is the entry point, but in the future
               if the entry point is different, we need to use that function id to calculate the gas.
            */
            for (func_id, costs_map) in metadata.gas_info.function_costs.iter() {
                if func_id.debug_name.as_ref().unwrap().as_str().ends_with("::main") {
                    let cost: usize = costs_map
                        .iter()
                        .map(|(token_type, cost)| {
                            let val_usize: usize = (*cost).try_into().unwrap();
                            // 10_000 is a dummy value for cost types that are not constant (same as in the casm runner)
                            let token_cost =
                                if *token_type == CostTokenType::Const { 1 } else { 10_000 };
                            val_usize * token_cost
                        })
                        .sum();
                    available_gas = if let Some(available_gas) = available_gas.checked_sub(cost) {
                        available_gas
                    } else {
                        bail!("not enough gas to call");
                    };
                    break;
                }
            }

            debug!("available gas after entry point spenditure = {}", available_gas);

            Some(CompilerGas { metadata, available_gas })
        } else {
            None
        };

        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();

        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        let location = Location::unknown(&context);

        let region = Region::new();
        let block = Block::new(&[]);
        region.append_block(block);
        let module_op = operation::Builder::new("builtin.module", location)
            /*
            .add_attributes(&[NamedAttribute::new_parsed(
                &context,
                "gpu.container_module",
                "unit",
            )
            .unwrap()])
            */
            .add_regions(vec![region])
            .build();

        let module = Module::from_operation(module_op).unwrap();

        Ok(Self {
            program,
            registry: sierra_program_registry,
            gas,
            context,
            module,
            main_print,
            print_fd,
        })
    }

    pub fn compile(&'ctx self) -> color_eyre::Result<OperationRef<'ctx>> {
        let mut storage = Storage::default();
        if self.gas.is_some() {
            self.create_gas_global()?;
        }
        self.process_types(&mut storage)?;
        self.process_libfuncs(&mut storage)?;
        self.process_functions(&mut storage)?;
        self.process_statements(&mut storage)?;
        Ok(self.module.as_operation())
    }

    pub fn compile_hardcoded_gpu(&'ctx self) -> color_eyre::Result<OperationRef<'ctx>> {
        /*
        fn main(a: i32, b: i32) -> i32 {
            a * b
        }
        */

        let i32_type = Type::integer(&self.context, 32);
        let i256_type = Type::integer(&self.context, 256);
        let index_type = Type::index(&self.context);
        let location = Location::unknown(&self.context);

        let gpu_module = {
            let module_region = Region::new();
            let module_block = Block::new(&[]);

            let region = Region::new();
            let block = Block::new(&[(i256_type, location), (i256_type, location)]);

            let arg1 = block.argument(0)?;
            let arg2 = block.argument(1)?;

            let res = self.op_add(&block, arg1.into(), arg2.into());
            let res_result = res.result(0)?;

            let trunc_op = block.append_operation(
                operation::Builder::new("arith.trunci", Location::unknown(&self.context))
                    .add_operands(&[res_result.into()])
                    .add_results(&[i32_type])
                    .build(),
            );

            let trunc_op_res = trunc_op.result(0)?;

            block.append_operation(
                operation::Builder::new("gpu.printf", Location::unknown(&self.context))
                    .add_attributes(&[self.named_attribute("format", r#""suma: %d ""#)?])
                    .add_operands(&[trunc_op_res.into()])
                    .build(),
            );

            // kernels always return void
            block.append_operation(
                operation::Builder::new("gpu.return", Location::unknown(&self.context)).build(),
            );

            region.append_block(block);

            let func = operation::Builder::new("gpu.func", Location::unknown(&self.context))
                .add_attributes(&NamedAttribute::new_parsed_vec(
                    &self.context,
                    &[
                        ("function_type", "(i256, i256) -> ()"),
                        ("sym_name", "\"kernel1\""),
                        ("gpu.kernel", "unit"),
                    ],
                )?)
                .add_regions(vec![region])
                .build();

            module_block.append_operation(func);

            module_block.append_operation(
                operation::Builder::new("gpu.module_end", Location::unknown(&self.context)).build(),
            );

            module_region.append_block(module_block);

            let gpu_module =
                operation::Builder::new("gpu.module", Location::unknown(&self.context))
                    .add_attributes(&[self.named_attribute("sym_name", "\"kernels\"")?])
                    .add_regions(vec![module_region])
                    .build();

            gpu_module
        };

        self.module.body().append_operation(gpu_module);

        let main_function = {
            let region = Region::new();
            let block = Block::new(&[]);

            let index_op = block.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_results(&[index_type])
                    .add_attributes(&[
                        self.named_attribute("value", &format!("1 : {}", index_type))?
                    ])
                    .build(),
            );
            let index_value = index_op.result(0)?.into();

            let dynamic_shared_memory_size_op = block.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_results(&[i32_type])
                    .add_attributes(&[self.named_attribute("value", &format!("0 : {}", i32_type))?])
                    .build(),
            );
            let dynamic_shared_memory_size = dynamic_shared_memory_size_op.result(0)?.into();

            let arg1_op = block.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_results(&[i256_type])
                    .add_attributes(
                        &[self.named_attribute("value", &format!("4 : {}", i256_type))?],
                    )
                    .build(),
            );
            let arg1 = arg1_op.result(0)?;
            let arg2_op = block.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_results(&[i256_type])
                    .add_attributes(
                        &[self.named_attribute("value", &format!("2 : {}", i256_type))?],
                    )
                    .build(),
            );
            let arg2 = arg2_op.result(0)?;

            let gpu_launch =
                operation::Builder::new("gpu.launch_func", Location::unknown(&self.context))
                    .add_attributes(&NamedAttribute::new_parsed_vec(
                        &self.context,
                        &[
                            ("kernel", "@kernels::@kernel1"),
                            ("operand_segment_sizes", "array<i32: 0, 1, 1, 1, 1, 1, 1, 1, 2>"),
                        ],
                    )?)
                    .add_operands(&[
                        index_value,
                        index_value,
                        index_value,
                        index_value,
                        index_value,
                        index_value,
                        dynamic_shared_memory_size,
                        arg1.into(),
                        arg2.into(),
                    ])
                    .build();

            block.append_operation(gpu_launch);

            let main_ret = self.op_const(&block, "0", self.i32_type());
            self.op_return(&block, &[main_ret.result(0)?.into()]);
            region.append_block(block);

            self.op_func(
                "main",
                "() -> i32",
                vec![region],
                FnAttributes { public: true, emit_c_interface: true, ..Default::default() },
            )?
        };

        self.module.body().append_operation(main_function);

        let op = self.module.as_operation();

        if op.verify() {
            Ok(op)
        } else {
            Err(color_eyre::eyre::eyre!("error verifiying"))
        }
    }
}
