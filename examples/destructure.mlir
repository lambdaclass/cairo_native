module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"struct_construct<destructure::destructure::MyStruct>"(%arg0: i256, %arg1: i256, %arg2: i256) -> !llvm.struct<(i256, i256, i256)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256, i256)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i256, i256, i256)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(i256, i256, i256)> 
    llvm.return %3 : !llvm.struct<(i256, i256, i256)>
  }
  llvm.func internal @"struct_deconstruct<destructure::destructure::MyStruct>"(%arg0: !llvm.struct<(i256, i256, i256)>) -> !llvm.struct<(i256, i256, i256)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i256, i256, i256)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i256, i256, i256)> 
    %2 = llvm.extractvalue %arg0[2] : !llvm.struct<(i256, i256, i256)> 
    %3 = llvm.mlir.undef : !llvm.struct<(i256, i256, i256)>
    %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(i256, i256, i256)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(i256, i256, i256)> 
    %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(i256, i256, i256)> 
    llvm.return %6 : !llvm.struct<(i256, i256, i256)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func internal @print_Unit(%arg0: !llvm.struct<()>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    llvm.return
  }
  llvm.func @main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"destructure::destructure::main"() : () -> !llvm.struct<()>
    llvm.call @print_Unit(%0) : (!llvm.struct<()>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"destructure::destructure::main"() -> !llvm.struct<()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(12 : i256) : i256
    %1 = llvm.mlir.constant(34 : i256) : i256
    %2 = llvm.mlir.constant(56 : i256) : i256
    %3 = llvm.call @"struct_construct<destructure::destructure::MyStruct>"(%0, %1, %2) : (i256, i256, i256) -> !llvm.struct<(i256, i256, i256)>
    %4 = llvm.call @"struct_deconstruct<destructure::destructure::MyStruct>"(%3) : (!llvm.struct<(i256, i256, i256)>) -> !llvm.struct<(i256, i256, i256)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i256, i256, i256)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(i256, i256, i256)> 
    %7 = llvm.extractvalue %4[2] : !llvm.struct<(i256, i256, i256)> 
    %8 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.return %8 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_destructure::destructure::main"(%arg0: !llvm.ptr<struct<()>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"destructure::destructure::main"() : () -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
}
