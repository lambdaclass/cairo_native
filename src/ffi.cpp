#include <llvm-c/Support.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>


// PLT: I need docs on this. I'm assuming Melior doesn't implement these because of a lack of a C API.
// If that's the case, a comment mentioning the last versions of both Melior and MLIR we checked that lacked it.
extern "C" const void *LLVMStructType_getFieldTypeAt(const void *ty_ptr, unsigned index)
{
    mlir::LLVM::LLVMStructType type = mlir::LLVM::LLVMStructType::getFromOpaquePointer(ty_ptr);

    return type.getBody()[index].getAsOpaquePointer();
}

extern "C" LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirOperation module,
                                          LLVMContextRef context) {
  mlir::Operation *moduleOp = unwrap(module);

  llvm::LLVMContext *ctx = llvm::unwrap(context);

  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(
      moduleOp, *ctx);

  LLVMModuleRef moduleRef = llvm::wrap(llvmModule.release());

  return moduleRef;
}
// PLT: ACK
