module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed ()>
    llvm.return %0 : !llvm.struct<packed ()>
  }
  llvm.func @"types::types::main"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    llvm.return %0 : !llvm.struct<packed ()>
  }
  llvm.func @"_mlir_ciface_types::types::main"(%arg0: !llvm.ptr<struct<packed ()>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"types::types::main"() : () -> !llvm.struct<packed ()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed ()>>
    llvm.return
  }
}
