//! # Code generation metadata
//!
//! The metadata is used for various stuff that would be otherwise difficult or plain impossible
//! with the current design, such as:
//!   - Pass compile-time constants that would otherwise have to be hardcoded in various places (ex.
//!     [PrimeModuloMeta](self::prime_modulo)).
//!   - Declare FFI bindings to external libraries (ex.
//!     [ReallocBindingsMeta](self::realloc_bindings)).
//!   - Pass extra compilation info to the libfunc generators (ex.
//!     [TailRecursionMeta](self::tail_recursion)).

use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

pub mod gas;
pub mod prime_modulo;
pub mod realloc_bindings;
pub mod runtime_bindings;
pub mod syscall_handler;
pub mod tail_recursion;

/// Metadata container.
#[derive(Default)]
pub struct MetadataStorage {
    entries: HashMap<TypeId, Box<dyn Any>>,
}

impl MetadataStorage {
    /// Create an empty metadata container.
    pub fn new() -> Self {
        Self {
            entries: HashMap::default(),
        }
    }

    /// Insert some metadata and return a mutable reference.
    ///
    /// The insertion will fail if there is already some metadata with the same type, in which case
    /// it'll return `None`.
    pub fn insert<T>(&mut self, meta: T) -> Option<&mut T>
    where
        T: Any,
    {
        if self.entries.contains_key(&TypeId::of::<T>()) {
            None
        } else {
            self.entries.insert(TypeId::of::<T>(), Box::new(meta));
            self.get_mut::<T>()
        }
    }

    /// Remove some metadata and return its last value.
    ///
    /// The removal will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Any,
    {
        self.entries
            .remove(&TypeId::of::<T>())
            .map(|meta| *(Box::<(dyn Any + 'static)>::downcast::<T>(meta).unwrap()))
    }

    /// Retrieve a reference to some metadata.
    ///
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn get<T>(&self) -> Option<&T>
    where
        T: Any,
    {
        self.entries
            .get(&TypeId::of::<T>())
            .map(|meta| meta.downcast_ref::<T>().unwrap())
    }

    /// Retrieve a mutable reference to some metadata.
    ///
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Any,
    {
        self.entries
            .get_mut(&TypeId::of::<T>())
            .map(|meta| meta.downcast_mut::<T>().unwrap())
    }
}

#[cfg(test)]
mod test {
    use super::{runtime_bindings::RuntimeBindingsMeta, *};

    #[test]
    fn runtime_library_insert_works() {
        let mut metadata = MetadataStorage::new();
        let ret = metadata.insert(RuntimeBindingsMeta::default());

        assert!(ret.is_some());
    }
}
