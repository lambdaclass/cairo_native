pub use self::{
    mappings::GraphMappings, memory::Memory, program::decode_instruction,
    search::run_search_algorithm, trace::Trace,
};

mod mappings;
mod memory;
mod program;
pub mod search;
mod trace;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct StepId(pub usize);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ValueId(pub usize);
