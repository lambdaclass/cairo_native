#![cfg(feature = "with-mem-tracing")]

use libc::{c_void, size_t};
use melior::ExecutionEngine;
use std::cell::UnsafeCell;

thread_local! {
    static MEM_TRACING: UnsafeCell<MemTracing> = const { UnsafeCell::new(MemTracing::new()) };
}

struct MemTracing {
    finished: Vec<AllocTrace>,
    pending: Vec<AllocTrace>,
}

struct AllocTrace {
    ptr: *mut c_void,
    len: size_t,
}

impl MemTracing {
    pub const fn new() -> Self {
        Self {
            finished: Vec::new(),
            pending: Vec::new(),
        }
    }

    pub fn push(&mut self, trace: AllocTrace) {
        match self.pending.binary_search_by_key(&trace.ptr, |x| x.ptr) {
            Ok(_) => unreachable!(),
            Err(pos) => self.pending.insert(pos, trace),
        }
    }

    pub fn update(&mut self, ptr: *mut c_void, trace: AllocTrace) {
        if let Ok(pos) = self.pending.binary_search_by_key(&ptr, |x| x.ptr) {
            let trace = self.pending.remove(pos);
            if trace.len == 0 {
                self.finished.push(trace);
                return;
            }
        };

        self.push(trace);
    }

    pub fn finish(&mut self, ptr: *mut c_void) {
        if let Ok(pos) = self.pending.binary_search_by_key(&ptr, |x| x.ptr) {
            let trace = self.pending.remove(pos);
            self.finished.push(trace);
        };
    }
}

impl AllocTrace {
    pub fn new(ptr: *mut c_void, len: size_t) -> Self {
        Self { ptr, len }
    }
}

pub(crate) fn register_bindings(engine: &ExecutionEngine) {
    unsafe {
        engine.register_symbol(
            "malloc",
            _wrapped_malloc as *const fn(size_t) -> *mut c_void as *mut (),
        );
        engine.register_symbol(
            "realloc",
            _wrapped_realloc as *const fn(*mut c_void, size_t) -> *mut c_void as *mut (),
        );
        engine.register_symbol("free", _wrapped_free as *const fn(*mut c_void) as *mut ());
    }
}

pub fn report_stats() {
    unsafe {
        // println!();
        // println!("[MemTracing] Stats:");
        // println!(
        //     "[MemTracing]   Freed allocations: {}",
        //     MEM_TRACING.finished.len()
        // );
        // println!("[MemTracing]   Pending allocations:");
        // for AllocTrace { ptr, len } in &MEM_TRACING.pending {
        //     println!("[MemTracing]     - {ptr:?} ({len} bytes)");
        // }

        MEM_TRACING.with(|x| {
            assert!((*x.get()).pending.is_empty());
            *x.get() = MemTracing::new();
        });
    }
}

pub(crate) unsafe extern "C" fn _wrapped_malloc(len: size_t) -> *mut c_void {
    let ptr = libc::malloc(len);

    println!("[MemTracing] Allocating ptr {ptr:?} with {len} bytes.");
    MEM_TRACING.with(|x| (*x.get()).push(AllocTrace::new(ptr, len)));

    ptr
}

pub(crate) unsafe extern "C" fn _wrapped_realloc(ptr: *mut c_void, len: size_t) -> *mut c_void {
    let new_ptr = libc::realloc(ptr, len);

    println!("[MemTracing] Reallocating {ptr:?} into {new_ptr:?} with {len} bytes.");
    MEM_TRACING.with(|x| (*x.get()).update(ptr, AllocTrace::new(new_ptr, len)));

    new_ptr
}

pub(crate) unsafe extern "C" fn _wrapped_free(ptr: *mut c_void) {
    libc::free(ptr);

    println!("[MemTracing] Freeing {ptr:?}.");
    MEM_TRACING.with(|x| (*x.get()).finish(ptr));
}
