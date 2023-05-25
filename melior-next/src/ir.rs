//! IR objects and builders.

mod affine_map;
mod attribute;
pub mod block;
mod identifier;
mod location;
mod module;
pub mod named_attribute;
pub mod operation;
mod region;
pub mod r#type;
mod value;

pub use self::{
    affine_map::AffineMap,
    attribute::Attribute,
    block::{Block, BlockRef},
    identifier::Identifier,
    location::Location,
    module::Module,
    named_attribute::NamedAttribute,
    operation::{Operation, OperationRef},
    r#type::{Type, TypeLike},
    region::{Region, RegionRef},
    value::{Value, ValueLike},
};
