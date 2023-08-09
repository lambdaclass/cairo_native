module attributes {llvm.data_layout = ""} {
  llvm.func @"program::program::felt_to_bool"(%arg0: i252) -> !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(18446744073709551613 : i256) : i256
    %1 = llvm.mlir.constant(4 : i256) : i256
    %2 = llvm.mlir.constant(0 : i252) : i252
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>>
    %5 = llvm.alloca %3 x !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>>
    %6 = llvm.zext %arg0 : i252 to i256
    %7 = llvm.sub %6, %1  : i256
    %8 = llvm.add %6, %0  : i256
    %9 = llvm.icmp "ult" %6, %1 : i256
    %10 = llvm.select %9, %8, %7 : i1, i256
    %11 = llvm.trunc %10 : i256 to i252
    %12 = llvm.icmp "eq" %11, %2 : i252
    %13 = llvm.select %12, %5, %4 : i1, !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>>
    %14 = llvm.mlir.undef : !llvm.struct<()>
    %15 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<()>)>
    %16 = llvm.insertvalue %12, %15[0] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %17 = llvm.insertvalue %14, %16[2] : !llvm.struct<(i1, array<0 x i8>, struct<()>)> 
    %18 = llvm.bitcast %13 : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    llvm.store %17, %18 {alignment = 1 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>>
    %19 = llvm.load %13 {alignment = 1 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>>
    llvm.return %19 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>
  }
  llvm.func @"_mlir_ciface_program::program::felt_to_bool"(%arg0: !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>>, %arg1: i252) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::felt_to_bool"(%arg1) : (i252) -> !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>>
    llvm.return
  }
  llvm.func @"program::program::main"() -> !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)> attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.mlir.constant(4 : i252) : i252
    %1 = llvm.mlir.constant(0 : i252) : i252
    %2 = llvm.call @"program::program::felt_to_bool"(%1) : (i252) -> !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>
    %3 = llvm.call @"program::program::felt_to_bool"(%0) : (i252) -> !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>
    %4 = llvm.mlir.undef : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>
    %5 = llvm.insertvalue %2, %4[0] : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)> 
    %6 = llvm.insertvalue %3, %5[1] : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)> 
    llvm.return %6 : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>
  }
  llvm.func @"_mlir_ciface_program::program::main"(%arg0: !llvm.ptr<struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %0 = llvm.call @"program::program::main"() : () -> !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>>
    llvm.return
  }
}

