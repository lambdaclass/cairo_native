#include <llvm-c/Support.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

using namespace mlir::LLVM;
using namespace mlir;

extern "C" void mlirModuleCleanup(MlirModule mod) {
  auto x = unwrap(mod);
  if (!x.getOps().empty()) {
    for (auto &op : x.getOps().begin()->getBlock()->getOperations()) {
      if (llvm::CastInfo<LLVMFuncOp, Operation>::isPossible(op)) {
        LLVMFuncOp x = llvm::CastInfo<LLVMFuncOp, Operation>::doCast(op);
        if (x.getSymName().starts_with("_mlir_ciface")) {
          x->setLoc(mlir::UnknownLoc::get(x->getContext()));
        }
      }
    }
  }
}
