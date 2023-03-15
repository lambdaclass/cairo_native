#ifndef SIERRA_OPS
#define SIERRA_OPS

include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SierraBase.td"

class Sierra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sierra_Dialect, mnemonic, traits>;

#endif
