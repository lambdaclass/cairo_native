use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

pub mod prime_modulo;
pub mod realloc_bindings;
pub mod tail_recursion;

pub struct MetadataStorage {
    entries: HashMap<TypeId, Box<dyn Any>>,
}

impl MetadataStorage {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::default(),
        }
    }

    pub fn insert<T>(&mut self, meta: T) -> Option<&mut T>
    where
        T: Any,
    {
        self.entries
            .try_insert(TypeId::of::<T>(), Box::new(meta))
            .ok()
            .map(|meta| meta.downcast_mut::<T>().unwrap())
    }

    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Any,
    {
        self.entries
            .remove(&TypeId::of::<T>())
            .map(|meta| Box::into_inner(Box::<(dyn Any + 'static)>::downcast::<T>(meta).unwrap()))
    }

    pub fn get<T>(&self) -> Option<&T>
    where
        T: Any,
    {
        self.entries
            .get(&TypeId::of::<T>())
            .map(|meta| meta.downcast_ref::<T>().unwrap())
    }

    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Any,
    {
        self.entries
            .get_mut(&TypeId::of::<T>())
            .map(|meta| meta.downcast_mut::<T>().unwrap())
    }
}
