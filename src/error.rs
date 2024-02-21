//! Various error types used thorough the crate.

pub use self::{
    builders::Error as BuilderError, compile::Error as CompileError,
    executor::Error as ExecutorError,
};

pub mod builders;
pub mod compile;
pub mod executor;
