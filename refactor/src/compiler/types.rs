use crate::ffi::Type;
use cairo_lang_sierra::ids::ConcreteTypeId;
use cxx::UniquePtr;
use std::{alloc::Layout, collections::BTreeMap, rc::Rc};

pub type TypeStorage = BTreeMap<SierraTypeId, Rc<CompiledType>>;

/// Compiled sierra type identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SierraTypeId(u64);

impl SierraTypeId {
    pub(super) const fn new(value: u64) -> Self {
        Self(value)
    }
}

/// A compiled Sierra type.
pub struct TypeLayout(pub(crate) Rc<CompiledType>);

/// A compiled Sierra type.
#[allow(dead_code)]
pub enum CompiledType {
    /// Integer types.
    Integer {
        id: Option<ConcreteTypeId>,
        mlir_type: UniquePtr<Type>,
        width: u32,
    },
    /// Structured types.
    Struct {
        id: Option<ConcreteTypeId>,
        mlir_type: UniquePtr<Type>,
        field_types: Vec<CompiledType>,
    },
    /// Dynamically allocated array.
    Vector {
        id: Option<ConcreteTypeId>,
        mlir_type: UniquePtr<Type>,
        inner: Box<CompiledType>,
    },

    Other {
        mlir_type: UniquePtr<Type>,
    },
}

#[allow(dead_code)]
impl CompiledType {
    pub fn id(&self) -> Option<&ConcreteTypeId> {
        match self {
            CompiledType::Vector { id, .. }
            | CompiledType::Integer { id, .. }
            | CompiledType::Struct { id, .. } => id.as_ref(),
            CompiledType::Other { .. } => unreachable!(),
        }
    }

    pub fn inner(&self) -> &CompiledType {
        match self {
            CompiledType::Vector { inner, .. } => inner,
            _ => panic!(),
        }
    }

    pub fn fields(&self) -> &[CompiledType] {
        match self {
            CompiledType::Struct { field_types, .. } => field_types,
            _ => panic!(),
        }
    }

    pub fn layout(&self) -> Layout {
        todo!()
    }

    pub(crate) fn mlir_type(&self) -> &Type {
        match self {
            CompiledType::Struct { mlir_type, .. }
            | CompiledType::Integer { mlir_type, .. }
            | CompiledType::Vector { mlir_type, .. }
            | CompiledType::Other { mlir_type } => mlir_type,
        }
    }
}
