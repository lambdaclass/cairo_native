use libc::{
    c_int, sigaction, sigaltstack, siginfo_t, sigset_t, stack_t, ucontext_t, SA_ONSTACK,
    SA_SIGINFO, SIGSEGV, SIGSTKSZ,
};
use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    ptr::{self, null_mut},
};
use thiserror::Error;

extern "C" {
    fn setjmp(env: *mut ()) -> c_int;
    fn longjmp(env: *mut (), val: c_int) -> !;
}

thread_local! {
    static STATE: UnsafeCell<SafeRunnerState> = const { UnsafeCell::new(SafeRunnerState::Inactive) };
    static STACK: UnsafeCell<SignalStack> = const { UnsafeCell::new(SignalStack(MaybeUninit::uninit())) };
}

type JmpBuf = MaybeUninit<[u8; 1024]>;

#[repr(align(16))]
#[allow(dead_code)]
struct SignalStack(MaybeUninit<[u8; SIGSTKSZ]>);

enum SafeRunnerState {
    Inactive,
    Active(Box<JmpBuf>),
}

#[derive(Debug, Error)]
pub enum SafeRunnerError {
    #[error("program execution aborted")]
    Aborted,
    #[error("program execution segfaulted")]
    Segfault,
}

/// Configure the current **process** for the [`SafeRunner`].
///
/// Note: It will override the previous signal handler for SIGSEGV.
pub fn setup_safe_runner() {
    unsafe {
        assert_eq!(
            sigaction(
                SIGSEGV,
                &sigaction {
                    sa_sigaction: segfault_handler
                        as *const extern "C" fn(c_int, &siginfo_t, &mut ucontext_t)
                        as usize,
                    sa_mask: MaybeUninit::<sigset_t>::zeroed().assume_init(),
                    sa_flags: SA_ONSTACK | SA_SIGINFO,
                    #[cfg(target_os = "linux")]
                    sa_restorer: None,
                },
                null_mut(),
            ),
            0,
        );
        assert_eq!(
            sigaltstack(
                &stack_t {
                    ss_sp: STACK.with(|x| x.get()).cast(),
                    ss_flags: 0,
                    ss_size: SIGSTKSZ,
                },
                null_mut(),
            ),
            0,
        );
    }
}

/// Manually trigger the segfault handler, thus aborting the current program.
pub fn abort_safe_runner() -> ! {
    unsafe {
        match STATE.with(|x| &mut *x.get()) {
            SafeRunnerState::Inactive => libc::abort(),
            SafeRunnerState::Active(jmp_buf) => longjmp(jmp_buf.as_mut_ptr().cast(), 2),
        }
    }
}

pub fn run_safely<T>(f: impl FnOnce() -> T) -> Result<T, SafeRunnerError> {
    let (jmp_buf, prev_state) = STATE.with(|x| unsafe {
        let jmp_buf;
        let prev_state = ptr::replace(
            x.get(),
            SafeRunnerState::Active({
                let mut tmp = Box::new(JmpBuf::uninit());
                jmp_buf = tmp.as_mut_ptr();
                tmp
            }),
        );

        (jmp_buf, prev_state)
    });

    let jmp_ret = unsafe { setjmp(jmp_buf.cast()) };
    let result = match jmp_ret {
        0 => Ok(f()),
        1 => Err(SafeRunnerError::Segfault),
        2 => Err(SafeRunnerError::Aborted),
        _ => unreachable!(),
    };

    STATE.with(|x| unsafe { ptr::write(x.get(), prev_state) });
    result
}

unsafe extern "C" fn segfault_handler(_sig: c_int, _info: &siginfo_t, _context: &mut ucontext_t) {
    match STATE.with(|x| &mut *x.get()) {
        SafeRunnerState::Inactive => libc::abort(),
        SafeRunnerState::Active(jmp_buf) => longjmp(jmp_buf.as_mut_ptr().cast(), 1),
    }
}
