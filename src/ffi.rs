use llvm_sys::{
    core::{
        LLVMContextCreate, LLVMContextDispose, LLVMDisposeMemoryBuffer, LLVMDisposeMessage,
        LLVMDisposeModule, LLVMGetBufferSize, LLVMGetBufferStart,
    },
    prelude::{LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef},
    target::{
        LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos, LLVM_InitializeAllTargetMCs,
        LLVM_InitializeAllTargets,
    },
    target_machine::{
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetMachine,
        LLVMDisposeTargetMachine, LLVMGetDefaultTargetTriple, LLVMGetHostCPUFeatures,
        LLVMGetHostCPUName, LLVMGetTargetFromTriple, LLVMRelocMode, LLVMTargetMachineEmitToFile,
        LLVMTargetMachineEmitToMemoryBuffer, LLVMTargetRef,
    },
};
use melior::ir::{Module, Type, TypeLike};
use mlir_sys::MlirOperation;
use std::{
    error::Error,
    ffi::{c_void, CStr, CString},
    fmt::Display,
    mem::MaybeUninit,
    ptr::{addr_of_mut, null_mut},
    sync::OnceLock,
};

extern "C" {
    fn LLVMStructType_getFieldTypeAt(ty_ptr: *const c_void, index: u32) -> *const c_void;

    /// Translate operation that satisfies LLVM dialect module requirements into an LLVM IR module living in the given context.
    /// This translates operations from any dilalect that has a registered implementation of LLVMTranslationDialectInterface.
    fn mlirTranslateModuleToLLVMIR(
        module_operation_ptr: MlirOperation,
        llvm_context: LLVMContextRef,
    ) -> LLVMModuleRef;
}

/// For any `!llvm.struct<...>` type, return the MLIR type of the field at the requested index.
pub fn get_struct_field_type_at<'c>(r#type: &Type<'c>, index: usize) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { LLVMStructType_getFieldTypeAt(ty_ptr.ptr, index as u32) };
    unsafe { Type::from_raw(ty_ptr) }
}

#[derive(Debug, Clone)]
pub struct LLVMCompileError(String);

impl Error for LLVMCompileError {}

impl Display for LLVMCompileError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Make sure to call
pub fn module_to_object(module: Module<'_>) -> Result<Vec<u8>, LLVMCompileError> {
    static INITIALIZED: OnceLock<()> = OnceLock::new();

    INITIALIZED.get_or_init(|| unsafe {
        LLVM_InitializeAllTargets();
        LLVM_InitializeAllTargetInfos();
        LLVM_InitializeAllTargetMCs();
        LLVM_InitializeAllAsmPrinters();
    });

    unsafe {
        let llvm_context = LLVMContextCreate();

        let op = module.as_operation().to_raw();

        let llvm_module = mlirTranslateModuleToLLVMIR(op, llvm_context);

        let mut null = null_mut();
        let mut error_buffer = addr_of_mut!(null);

        let target_triple = LLVMGetDefaultTargetTriple();
        let target_cpu = LLVMGetHostCPUName();
        let target_cpu_features = LLVMGetHostCPUFeatures();

        let mut target: MaybeUninit<LLVMTargetRef> = MaybeUninit::uninit();

        if LLVMGetTargetFromTriple(target_triple, target.as_mut_ptr(), error_buffer) != 0 {
            let error = CStr::from_ptr(*error_buffer);
            let err = error.to_string_lossy().to_string();
            LLVMDisposeMessage(*error_buffer);
            Err(LLVMCompileError(err))?;
        } else if !(*error_buffer).is_null() {
            LLVMDisposeMessage(*error_buffer);
            error_buffer = addr_of_mut!(null);
        }

        let target = target.assume_init();

        let machine = LLVMCreateTargetMachine(
            target,
            target_triple.cast(),
            target_cpu.cast(),
            target_cpu_features.cast(),
            LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            LLVMRelocMode::LLVMRelocDefault,
            LLVMCodeModel::LLVMCodeModelDefault,
        );

        let mut out_buf: MaybeUninit<LLVMMemoryBufferRef> = MaybeUninit::uninit();

        let ok = LLVMTargetMachineEmitToMemoryBuffer(
            machine,
            llvm_module,
            LLVMCodeGenFileType::LLVMObjectFile,
            error_buffer,
            out_buf.as_mut_ptr(),
        );

        if ok != 0 {
            let error = CStr::from_ptr(*error_buffer);
            let err = error.to_string_lossy().to_string();
            LLVMDisposeMessage(*error_buffer);
            Err(LLVMCompileError(err))?;
        } else if !(*error_buffer).is_null() {
            LLVMDisposeMessage(*error_buffer);
            error_buffer = addr_of_mut!(null);
        }

        let out_buf = out_buf.assume_init();

        let out_buf_start: *const u8 = LLVMGetBufferStart(out_buf).cast();
        let out_buf_size = LLVMGetBufferSize(out_buf);

        // keep it in rust side
        let data = std::slice::from_raw_parts(out_buf_start, out_buf_size).to_vec();

        LLVMDisposeMemoryBuffer(out_buf);
        LLVMDisposeTargetMachine(machine);
        LLVMDisposeModule(llvm_module);
        LLVMContextDispose(llvm_context);

        Ok(data)
    }
}

pub fn object_to_shared_lib(object: &[u8]) -> &[u8] {
    /*
    let llvm_location = PathBuf::from(std::env::var("MLIR_SYS_170_PREFIX").unwrap());
    let clang_location = llvm_location.join("bin").join("clang");

    let mut clang = std::process::Command::new(clang_location);
    let mut proc = clang.args(["--shared", "-x assembler", "-Wno-override-module", "-", "-o -"]).spawn().unwrap();
    proc.stdin.as_mut().unwrap().write_all(data).unwrap();
    let output = proc.wait_with_output().unwrap();

    std::fs::write("output.s", output.stdout).unwrap();
    */
    todo!()
}
