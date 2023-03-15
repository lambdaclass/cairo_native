// Include the definition of the necessary tablegen constructs for defining
// our dialect.

#ifndef SIERRA_BASE
#define SIERRA_BASE

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Here is a simple definition of a dialect.
def Sierra_Dialect : Dialect {
  let name = "sierra";
  let cppNamespace = "::sierra";
  let summary = "Sierra dialect";

  let description = [{
    Currently just experimenting
  }];

  let dependentDialects = ["arith::ArithDialect"];
}

#endif
