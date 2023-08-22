//! Various error types used thorough the crate.

pub use self::{
    compile::Error as CompileError, jit_engine::Error as JitRunnerError,
    libfuncs::Error as CoreLibfuncBuilderError, types::Error as CoreTypeBuilderError,
};
pub mod compile;
pub mod jit_engine;
pub mod libfuncs;
pub mod types;
