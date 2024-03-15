#include <llvm-c/Support.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/IR/Types.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>


extern "C" const void *LLVMStructType_getFieldTypeAt(const void *ty_ptr, unsigned index)
{
    mlir::LLVM::LLVMStructType type = mlir::LLVM::LLVMStructType::getFromOpaquePointer(ty_ptr);

    return type.getBody()[index].getAsOpaquePointer();
}

extern "C" uint64_t DataLayout_getTypeABIAlignment(MlirOperation module, MlirType type)
{
    mlir::Operation *moduleOp = unwrap(module);
    mlir::Type typeInfo = unwrap(type);

    mlir::DataLayout dataLayout(moduleOp->getParentOfType<mlir::DataLayoutOpInterface>());
    return dataLayout.getTypeABIAlignment(typeInfo);
}

extern "C" uint64_t DataLayout_getTypeSize(MlirOperation module, MlirType type)
{
    mlir::Operation *moduleOp = unwrap(module);
    mlir::Type typeInfo = unwrap(type);

    mlir::DataLayout dataLayout(moduleOp->getParentOfType<mlir::DataLayoutOpInterface>());
    return dataLayout.getTypeSize(typeInfo);
}


extern "C" LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirOperation module, LLVMContextRef context)
{
    llvm::LLVMContext *ctx = llvm::unwrap(context);
    mlir::Operation *moduleOp = unwrap(module);

    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(moduleOp, *ctx);
    return llvm::wrap(llvmModule.release());
}
