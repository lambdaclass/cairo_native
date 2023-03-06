//! A MLIR safe API wrapper
//!

//#![deny(warnings)]
//#![deny(clippy::nursery)]

pub mod block;
pub mod context;
pub mod dialects;
pub mod location;
pub mod mlir_type;
pub mod module;
pub mod registry;
pub mod operation;
pub mod attribute;
pub mod identifier;
pub mod llvm_string;
