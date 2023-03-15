/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::sierra::SierraDialect)
namespace sierra {

SierraDialect::SierraDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<SierraDialect>()) {
  
    getContext()->loadDialect<arith::ArithDialect>();

  initialize();
}

SierraDialect::~SierraDialect() = default;

} // namespace sierra
