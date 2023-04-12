module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func internal @print_Unit(%arg0: !llvm.struct<()>) {
    llvm.return
  }
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"types::types::main"() : () -> !llvm.struct<()>
    llvm.call @print_Unit(%0) : (!llvm.struct<()>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"types::types::main"() -> !llvm.struct<()> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(123 : i8) : i8
    %1 = llvm.mlir.constant(123 : i16) : i16
    %2 = llvm.mlir.constant(123 : i32) : i32
    %3 = llvm.mlir.constant(123 : i64) : i64
    %4 = llvm.mlir.constant(123 : i128) : i128
    %5 = llvm.mlir.constant(123 : i256) : i256
    %6 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.return %6 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_types::types::main"(%arg0: !llvm.ptr<struct<()>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"types::types::main"() : () -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
}
