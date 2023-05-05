#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"


void test()
{
    mlir::MLIRContext c;
    mlir::OpBuilder b(&c);

    mlir::Region r;
    mlir::Block &block = *r.getBlocks().begin();

    mlir::OperationState state(b.getUnknownLoc(), "");
    state.addAttribute("", mlir::TypeAttr::get(b.getNoneType()));
}
