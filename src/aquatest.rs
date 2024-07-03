use aquamarine::aquamarine;

#![cfg_attr(doc, aquamarine)]
//! `AquaTest` is a very documented module.
//!
//! ```mermaid
//! stateDiagram-v2
//!     direction LR
//!     state cairo
//!     state sierra
//!     state casm
//!     state "cairo-vm" as cairovm
//!     cairo --> sierra
//!     sierra --> casm
//!     sierra --> mlir
//!     casm --> cairovm   
//!     state "cairo-native" as caironative {
//!         state mlir
//!         state llvm
//!         state object
//!         mlir --> llvm
//!         llvm --> object
//!         }
//! ```


#[cfg_attr(doc, aquamarine)]
/// `AquaTest` is a very documented type.
///
/// ```mermaid
/// stateDiagram-v2
///     direction LR
///     state cairo
///     state sierra
///     state casm
///     state "cairo-vm" as cairovm
///     cairo --> sierra
///     sierra --> casm
///     sierra --> mlir
///     casm --> cairovm   
///     state "cairo-native" as caironative {
///         state mlir
///         state llvm
///         state object
///         mlir --> llvm
///         llvm --> object
///         }
/// ```
pub struct AquaTest {
    f: i32
}
