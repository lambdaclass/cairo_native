use libc::{
    c_int, sigaction, sigaltstack, siginfo_t, sigset_t, stack_t, ucontext_t, SA_ONSTACK,
    SA_SIGINFO, SIGSEGV, SIGSTKSZ,
};
use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    ptr::{self, null_mut},
};

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
struct SignalStack(MaybeUninit<[u8; SIGSTKSZ]>);

enum SafeRunnerState {
    Inactive,
    Active(Box<JmpBuf>),
}

#[allow(clippy::result_unit_err)]
pub fn setup_safe_runner() -> Result<(), ()> {
    unsafe {
        let ret = sigaction(
            SIGSEGV,
            &sigaction {
                sa_sigaction: segfault_handler
                    as *const extern "C" fn(c_int, &siginfo_t, &mut ucontext_t)
                    as usize,
                sa_mask: MaybeUninit::<sigset_t>::zeroed().assume_init(),
                sa_flags: SA_ONSTACK | SA_SIGINFO,
                sa_restorer: None,
            },
            null_mut(),
        );
        if ret < 0 {
            return Err(());
        }

        sigaltstack(
            &stack_t {
                ss_sp: STACK.with(|x| x.get()).cast(),
                ss_flags: 0,
                ss_size: SIGSTKSZ,
            },
            null_mut(),
        );
    }

    Ok(())
}

pub fn run_safely<T>(f: impl FnOnce() -> T) -> Result<T, ()> {
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
        _ => Err(()),
    };

    STATE.with(|x| unsafe { ptr::write(x.get(), prev_state) });
    result
}

unsafe extern "C" fn segfault_handler(_sig: c_int, _info: &siginfo_t, _context: &mut ucontext_t) {
    match STATE.with(|x| &mut *x.get()) {
        SafeRunnerState::Inactive => libc::abort(),
        SafeRunnerState::Active(jmp_buf) => longjmp(jmp_buf.as_mut_ptr().cast(), 0),
    }
}
