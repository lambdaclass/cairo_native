//! Sierra to MLIR compiler.

#![deny(clippy::pedantic)]
#![deny(missing_docs)]
#![deny(warnings)]
#![allow(clippy::missing_panics_doc)]
#![allow(unused)]

pub mod compiler;
pub mod database;
mod ffi;
pub mod libfuncs;
pub mod types;
