module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @bitwise(%arg0: i128, %arg1: i128) -> !llvm.struct<(i128, i128, i128)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.and %arg0, %arg1  : i128
    %1 = llvm.xor %arg0, %arg1  : i128
    %2 = llvm.or %arg0, %arg1  : i128
    %3 = llvm.mlir.undef : !llvm.struct<(i128, i128, i128)>
    %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(i128, i128, i128)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(i128, i128, i128)> 
    %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(i128, i128, i128)> 
    llvm.return %6 : !llvm.struct<(i128, i128, i128)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed ()>
    llvm.return %0 : !llvm.struct<packed ()>
  }
  llvm.func internal @print_Unit(%arg0: !llvm.struct<packed ()>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    llvm.return
  }
  llvm.func @main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"bitwise::bitwise::main"() : () -> !llvm.struct<packed ()>
    llvm.call @print_Unit(%0) : (!llvm.struct<packed ()>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"bitwise::bitwise::main"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(1234 : i128) : i128
    %1 = llvm.mlir.constant(5678 : i128) : i128
    %2 = llvm.call @bitwise(%0, %1) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(i128, i128, i128)> 
    %4 = llvm.extractvalue %2[1] : !llvm.struct<(i128, i128, i128)> 
    %5 = llvm.extractvalue %2[2] : !llvm.struct<(i128, i128, i128)> 
    %6 = llvm.call @bitwise(%0, %1) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(i128, i128, i128)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(i128, i128, i128)> 
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(i128, i128, i128)> 
    %10 = llvm.call @bitwise(%0, %1) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(i128, i128, i128)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(i128, i128, i128)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(i128, i128, i128)> 
    %14 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    llvm.return %14 : !llvm.struct<packed ()>
  }
  llvm.func @"_mlir_ciface_bitwise::bitwise::main"(%arg0: !llvm.ptr<struct<packed ()>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"bitwise::bitwise::main"() : () -> !llvm.struct<packed ()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed ()>>
    llvm.return
  }
}
