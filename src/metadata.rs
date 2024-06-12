////! # Code generation metadata
//! # Code generation metadata
////!
//!
////! The metadata is used for various stuff that would be otherwise difficult or plain impossible
//! The metadata is used for various stuff that would be otherwise difficult or plain impossible
////! with the current design, such as:
//! with the current design, such as:
////!   - Pass compile-time constants that would otherwise have to be hardcoded in various places (ex.
//!   - Pass compile-time constants that would otherwise have to be hardcoded in various places (ex.
////!     [PrimeModuloMeta](self::prime_modulo)).
//!     [PrimeModuloMeta](self::prime_modulo)).
////!   - Declare FFI bindings to external libraries (ex.
//!   - Declare FFI bindings to external libraries (ex.
////!     [ReallocBindingsMeta](self::realloc_bindings)).
//!     [ReallocBindingsMeta](self::realloc_bindings)).
////!   - Pass extra compilation info to the libfunc generators (ex.
//!   - Pass extra compilation info to the libfunc generators (ex.
////!     [TailRecursionMeta](self::tail_recursion)).
//!     [TailRecursionMeta](self::tail_recursion)).
//

//use std::{
use std::{
//    any::{Any, TypeId},
    any::{Any, TypeId},
//    collections::{hash_map::Entry, HashMap},
    collections::{hash_map::Entry, HashMap},
//};
};
//

//pub mod debug_utils;
pub mod debug_utils;
//pub mod enum_snapshot_variants;
pub mod enum_snapshot_variants;
//pub mod gas;
pub mod gas;
//pub mod prime_modulo;
pub mod prime_modulo;
//pub mod realloc_bindings;
pub mod realloc_bindings;
//pub mod runtime_bindings;
pub mod runtime_bindings;
//pub mod snapshot_clones;
pub mod snapshot_clones;
//pub mod tail_recursion;
pub mod tail_recursion;
//

///// Metadata container.
/// Metadata container.
//#[cfg_attr(not(feature = "with-debug-utils"), derive(Default))]
#[cfg_attr(not(feature = "with-debug-utils"), derive(Default))]
//#[derive(Debug)]
#[derive(Debug)]
//pub struct MetadataStorage {
pub struct MetadataStorage {
//    entries: HashMap<TypeId, Box<dyn Any>>,
    entries: HashMap<TypeId, Box<dyn Any>>,
//}
}
//

//impl MetadataStorage {
impl MetadataStorage {
//    /// Create an empty metadata container.
    /// Create an empty metadata container.
//    pub fn new() -> Self {
    pub fn new() -> Self {
//        Self::default()
        Self::default()
//    }
    }
//

//    /// Insert some metadata and return a mutable reference.
    /// Insert some metadata and return a mutable reference.
//    ///
    ///
//    /// The insertion will fail if there is already some metadata with the same type, in which case
    /// The insertion will fail if there is already some metadata with the same type, in which case
//    /// it'll return `None`.
    /// it'll return `None`.
//    pub fn insert<T>(&mut self, meta: T) -> Option<&mut T>
    pub fn insert<T>(&mut self, meta: T) -> Option<&mut T>
//    where
    where
//        T: Any,
        T: Any,
//    {
    {
//        if let Entry::Vacant(e) = self.entries.entry(TypeId::of::<T>()) {
        if let Entry::Vacant(e) = self.entries.entry(TypeId::of::<T>()) {
//            e.insert(Box::new(meta));
            e.insert(Box::new(meta));
//            self.get_mut::<T>()
            self.get_mut::<T>()
//        } else {
        } else {
//            None
            None
//        }
        }
//    }
    }
//

//    /// Remove some metadata and return its last value.
    /// Remove some metadata and return its last value.
//    ///
    ///
//    /// The removal will fail if there is no metadata with the requested type, in which case it'll
    /// The removal will fail if there is no metadata with the requested type, in which case it'll
//    /// return `None`.
    /// return `None`.
//    pub fn remove<T>(&mut self) -> Option<T>
    pub fn remove<T>(&mut self) -> Option<T>
//    where
    where
//        T: Any,
        T: Any,
//    {
    {
//        self.entries
        self.entries
//            .remove(&TypeId::of::<T>())
            .remove(&TypeId::of::<T>())
//            .map(|meta| *(Box::<(dyn Any + 'static)>::downcast::<T>(meta).unwrap()))
            .map(|meta| *(Box::<(dyn Any + 'static)>::downcast::<T>(meta).unwrap()))
//    }
    }
//

//    /// Retrieve a reference to some metadata.
    /// Retrieve a reference to some metadata.
//    ///
    ///
//    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
//    /// return `None`.
    /// return `None`.
//    pub fn get<T>(&self) -> Option<&T>
    pub fn get<T>(&self) -> Option<&T>
//    where
    where
//        T: Any,
        T: Any,
//    {
    {
//        self.entries
        self.entries
//            .get(&TypeId::of::<T>())
            .get(&TypeId::of::<T>())
//            .map(|meta| meta.downcast_ref::<T>().unwrap())
            .map(|meta| meta.downcast_ref::<T>().unwrap())
//    }
    }
//

//    /// Retrieve a mutable reference to some metadata.
    /// Retrieve a mutable reference to some metadata.
//    ///
    ///
//    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
//    /// return `None`.
    /// return `None`.
//    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
//    where
    where
//        T: Any,
        T: Any,
//    {
    {
//        self.entries
        self.entries
//            .get_mut(&TypeId::of::<T>())
            .get_mut(&TypeId::of::<T>())
//            .map(|meta| meta.downcast_mut::<T>().unwrap())
            .map(|meta| meta.downcast_mut::<T>().unwrap())
//    }
    }
//

//    pub fn get_or_insert_with<T>(&mut self, meta_gen: impl FnOnce() -> T) -> &mut T
    pub fn get_or_insert_with<T>(&mut self, meta_gen: impl FnOnce() -> T) -> &mut T
//    where
    where
//        T: Any,
        T: Any,
//    {
    {
//        self.entries
        self.entries
//            .entry(TypeId::of::<T>())
            .entry(TypeId::of::<T>())
//            .or_insert_with(|| Box::new(meta_gen()))
            .or_insert_with(|| Box::new(meta_gen()))
//            .downcast_mut::<T>()
            .downcast_mut::<T>()
//            .unwrap()
            .unwrap()
//    }
    }
//}
}
//

//#[cfg(feature = "with-debug-utils")]
#[cfg(feature = "with-debug-utils")]
//impl Default for MetadataStorage {
impl Default for MetadataStorage {
//    fn default() -> Self {
    fn default() -> Self {
//        let mut metadata = Self {
        let mut metadata = Self {
//            entries: Default::default(),
            entries: Default::default(),
//        };
        };
//

//        metadata.insert(debug_utils::DebugUtils::default());
        metadata.insert(debug_utils::DebugUtils::default());
//

//        metadata
        metadata
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use super::{runtime_bindings::RuntimeBindingsMeta, *};
    use super::{runtime_bindings::RuntimeBindingsMeta, *};
//

//    #[test]
    #[test]
//    fn runtime_library_insert_works() {
    fn runtime_library_insert_works() {
//        let mut metadata = MetadataStorage::new();
        let mut metadata = MetadataStorage::new();
//        let ret = metadata.insert(RuntimeBindingsMeta::default());
        let ret = metadata.insert(RuntimeBindingsMeta::default());
//

//        assert!(ret.is_some());
        assert!(ret.is_some());
//    }
    }
//}
}
