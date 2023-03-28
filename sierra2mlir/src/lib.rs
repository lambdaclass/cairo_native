//! A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

//#![deny(warnings)]
#![warn(clippy::nursery)]
//#![deny(clippy::pedantic)]
#![warn(clippy::all)]
#![warn(unused)]

use melior_next::{pass, utility::register_all_passes, ExecutionEngine};
use tracing::debug;

use crate::compiler::Compiler;

pub mod compiler;
mod libfuncs;
mod statements;
pub mod types;
mod utility;

pub fn compile(
    code: &str,
    optimized: bool,
    debug_info: bool,
    main_print: bool,
) -> Result<String, color_eyre::Report> {
    let mut compiler = Compiler::new(code, main_print)?;

    if main_print {
        compiler.create_printf()?;
    }

    compiler.compile()?;

    debug!("mlir before pass:\n{}", compiler.module.as_operation());
    let pass_manager = pass::Manager::new(&compiler.context);
    register_all_passes();

    if optimized {
        // the inliner sometimes slows things down, need to investigate
        // pass_manager.add_pass(pass::transform::inliner());
        pass_manager.add_pass(pass::transform::sccp());
        pass_manager.add_pass(pass::transform::cse());
        pass_manager.add_pass(pass::transform::symbol_dce());
        pass_manager.add_pass(pass::transform::canonicalizer());
    }

    pass_manager.add_pass(pass::conversion::convert_scf_to_cf());
    pass_manager.add_pass(pass::conversion::convert_cf_to_llvm());
    //pass_manager.add_pass(pass::conversion::convert_gpu_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_arithmetic_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_func_to_llvm());

    // pass_manager.add_pass(pass::transform::print_operation_stats());
    pass_manager.enable_verifier(true);
    pass_manager.run(&mut compiler.module)?;

    let op = compiler.module.as_operation();
    if op.verify() {
        if debug_info {
            Ok(op.debug_print())
        } else {
            Ok(op.to_string())
        }
    } else {
        Err(color_eyre::eyre::eyre!("error verifiying"))
    }
}

pub fn execute(code: &str) -> Result<ExecutionEngine, color_eyre::Report> {
    let mut compiler = Compiler::new(code, false)?;
    compiler.compile()?;

    let pass_manager = pass::Manager::new(&compiler.context);
    register_all_passes();
    pass_manager.add_pass(pass::conversion::convert_scf_to_cf());
    pass_manager.add_pass(pass::conversion::convert_cf_to_llvm());
    //pass_manager.add_pass(pass::conversion::convert_gpu_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_func_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_arithmetic_to_llvm());
    pass_manager.enable_verifier(true);
    pass_manager.run(&mut compiler.module)?;

    let engine = ExecutionEngine::new(&compiler.module, 2, &[], false);

    Ok(engine)
}
