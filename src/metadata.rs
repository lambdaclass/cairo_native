use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::Debug,
};

pub mod tail_recursion;

pub struct MetadataStorage {
    entries: HashMap<TypeId, Box<dyn Metadata>>,
}

impl MetadataStorage {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::default(),
        }
    }

    pub fn insert<T>(&mut self, meta: T) -> Option<&mut T>
    where
        T: Any + Debug,
    {
        self.entries
            .try_insert(TypeId::of::<T>(), Box::new(meta))
            .ok()
            .map(|meta| (meta.as_mut() as &mut dyn Any).downcast_mut::<T>().unwrap())
    }

    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Any + Debug,
    {
        self.entries
            .remove(&TypeId::of::<T>())
            .map(|meta| Box::into_inner(Box::<(dyn Any + 'static)>::downcast::<T>(meta).unwrap()))
    }

    pub fn get<T>(&self) -> Option<&T>
    where
        T: Any + Debug,
    {
        self.entries
            .get(&TypeId::of::<T>())
            .map(|meta| (meta as &dyn Any).downcast_ref::<T>().unwrap())
    }

    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Any + Debug,
    {
        self.entries
            .get_mut(&TypeId::of::<T>())
            .map(|meta| (meta as &mut dyn Any).downcast_mut::<T>().unwrap())
    }
}

trait Metadata
where
    Self: Any + Debug,
{
}

impl<T> Metadata for T where T: 'static + Any + Debug {}
