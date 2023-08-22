//! Various error types used thorough the crate.

pub use self::{
    compile::Error as CompileError, jit_engine::Error as JitRunnerError,
    libfuncs::Error as CoreLibfuncBuilderError, types::Error as CoreTypeBuilderError,
};
pub mod compile;
pub mod jit_engine;
pub mod libfuncs;
pub mod types;

// #[derive(Error)]
// pub enum NativeError<'de, D, S>
// where
//     D: Deserializer<'de>,
//     S: Serializer,
// {
//     RunnerError(JitRunnerError<'de, CoreType, CoreLibfunc, D, S>),
//     CompilerError(CompileError<CoreType, CoreLibfunc>),
// }

// impl<'de, D, S> fmt::Debug for NativeError<'de, D, S>
// where
//     D: Deserializer<'de>,
//     S: Serializer,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             Self::CompilerError(x) => fmt::Debug::fmt(x, f),
//             Self::RunnerError(x) => fmt::Debug::fmt(x, f),
//         }
//     }
// }

// impl<'de, D, S> fmt::Display for NativeError<'de, D, S>
// where
//     D: Deserializer<'de>,
//     S: Serializer,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             Self::CompilerError(x) => fmt::Display::fmt(x, f),
//             Self::RunnerError(x) => fmt::Display::fmt(x, f),
//         }
//     }
// }

// impl<'de, D, S> From<crate::error::CompileError<CoreType, CoreLibfunc>> for NativeError<'de, D, S>
// where
//     D: Deserializer<'de>,
//     S: Serializer,
// {
//     fn from(value: crate::error::CompileError<CoreType, CoreLibfunc>) -> Self {
//         Self::CompilerError(value)
//     }
// }

// impl<'de, D, S> From<crate::error::JitRunnerError<'de, CoreType, CoreLibfunc, D, S>>
//     for NativeError<'de, D, S>
// where
//     D: Deserializer<'de>,
//     S: Serializer,
// {
//     fn from(value: crate::error::JitRunnerError<'de, CoreType, CoreLibfunc, D, S>) -> Self {
//         Self::RunnerError(value)
//     }
// }
