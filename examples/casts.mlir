module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"upcast<u8, u16>"(%arg0: i8) -> i16 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i16
    llvm.return %0 : i16
  }
  llvm.func internal @"upcast<u8, u32>"(%arg0: i8) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i32
    llvm.return %0 : i32
  }
  llvm.func internal @"upcast<u8, u64>"(%arg0: i8) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u8, u128>"(%arg0: i8) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u16, u32>"(%arg0: i16) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i32
    llvm.return %0 : i32
  }
  llvm.func internal @"upcast<u16, u64>"(%arg0: i16) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u16, u128>"(%arg0: i16) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u32, u64>"(%arg0: i32) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i32 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u32, u128>"(%arg0: i32) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i32 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u64, u128>"(%arg0: i64) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i64 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u8>, 0>"(%arg0: i8) -> !llvm.struct<packed (i16, array<1 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<1 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i8)>
    llvm.store %arg0, %4 : i8, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<1 x i8>)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed ()>
    llvm.return %0 : !llvm.struct<packed ()>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u8>, 1>"(%arg0: !llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<1 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<1 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed ()>)>
    llvm.store %arg0, %4 : !llvm.struct<packed ()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<1 x i8>)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u16>, 0>"(%arg0: i16) -> !llvm.struct<packed (i16, array<2 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<2 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i16)>
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<2 x i8>)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u16>, 1>"(%arg0: !llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<2 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<2 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed ()>)>
    llvm.store %arg0, %4 : !llvm.struct<packed ()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<2 x i8>)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%arg0: i32) -> !llvm.struct<packed (i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i32)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<4 x i8>)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%arg0: !llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed ()>)>
    llvm.store %arg0, %4 : !llvm.struct<packed ()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<4 x i8>)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u64>, 0>"(%arg0: i64) -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i64)>
    llvm.store %arg0, %4 : i64, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u64>, 1>"(%arg0: !llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed ()>)>
    llvm.store %arg0, %4 : !llvm.struct<packed ()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u128>, 0>"(%arg0: i128) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i128)>
    llvm.store %arg0, %4 : i128, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u128>, 1>"(%arg0: !llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed ()>)>
    llvm.store %arg0, %4 : !llvm.struct<packed ()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @print_Unit(%arg0: !llvm.struct<packed ()>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    llvm.return
  }
  llvm.func @main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"casts::casts::main"() : () -> !llvm.struct<packed ()>
    llvm.call @print_Unit(%0) : (!llvm.struct<packed ()>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"casts::casts::main"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb16
  ^bb1:  // pred: ^bb16
    %0 = llvm.mlir.constant(0 : i8) : i8
    llvm.br ^bb17(%16, %20, %23, %25, %0 : i16, i32, i64, i128, i8)
  ^bb2:  // pred: ^bb19
    %1 = llvm.trunc %39 : i16 to i8
    llvm.br ^bb20(%39, %40, %41, %42, %1 : i16, i32, i64, i128, i8)
  ^bb3:  // pred: ^bb22
    llvm.br ^bb23(%58, %59, %60, %57 : i32, i64, i128, i16)
  ^bb4:  // pred: ^bb25
    %2 = llvm.trunc %72 : i32 to i8
    llvm.br ^bb26(%72, %73, %74, %2 : i32, i64, i128, i8)
  ^bb5:  // pred: ^bb28
    %3 = llvm.trunc %87 : i32 to i16
    llvm.br ^bb29(%87, %88, %89, %3 : i32, i64, i128, i16)
  ^bb6:  // pred: ^bb31
    llvm.br ^bb32(%103, %104, %102 : i64, i128, i32)
  ^bb7:  // pred: ^bb34
    %4 = llvm.trunc %114 : i64 to i8
    llvm.br ^bb35(%114, %115, %4 : i64, i128, i8)
  ^bb8:  // pred: ^bb37
    %5 = llvm.trunc %126 : i64 to i16
    llvm.br ^bb38(%126, %127, %5 : i64, i128, i16)
  ^bb9:  // pred: ^bb40
    %6 = llvm.trunc %138 : i64 to i32
    llvm.br ^bb41(%138, %139, %6 : i64, i128, i32)
  ^bb10:  // pred: ^bb43
    llvm.br ^bb44(%151, %150 : i128, i64)
  ^bb11:  // pred: ^bb46
    %7 = llvm.trunc %159 : i128 to i8
    llvm.br ^bb47(%159, %7 : i128, i8)
  ^bb12:  // pred: ^bb49
    %8 = llvm.trunc %168 : i128 to i16
    llvm.br ^bb50(%168, %8 : i128, i16)
  ^bb13:  // pred: ^bb52
    %9 = llvm.trunc %177 : i128 to i32
    llvm.br ^bb53(%177, %9 : i128, i32)
  ^bb14:  // pred: ^bb55
    %10 = llvm.trunc %186 : i128 to i64
    llvm.br ^bb56(%186, %10 : i128, i64)
  ^bb15:  // pred: ^bb58
    llvm.br ^bb59(%195 : i128)
  ^bb16:  // pred: ^bb0
    %11 = llvm.mlir.constant(0 : i8) : i8
    %12 = llvm.call @"upcast<u8, u16>"(%11) : (i8) -> i16
    %13 = llvm.call @"upcast<u8, u32>"(%11) : (i8) -> i32
    %14 = llvm.call @"upcast<u8, u64>"(%11) : (i8) -> i64
    %15 = llvm.call @"upcast<u8, u128>"(%11) : (i8) -> i128
    %16 = llvm.mlir.constant(0 : i16) : i16
    %17 = llvm.call @"upcast<u16, u32>"(%16) : (i16) -> i32
    %18 = llvm.call @"upcast<u16, u64>"(%16) : (i16) -> i64
    %19 = llvm.call @"upcast<u16, u128>"(%16) : (i16) -> i128
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.call @"upcast<u32, u64>"(%20) : (i32) -> i64
    %22 = llvm.call @"upcast<u32, u128>"(%20) : (i32) -> i128
    %23 = llvm.mlir.constant(0 : i64) : i64
    %24 = llvm.call @"upcast<u64, u128>"(%23) : (i64) -> i128
    %25 = llvm.mlir.constant(0 : i128) : i128
    %26 = llvm.mlir.constant(true) : i1
    llvm.cond_br %26, ^bb1, ^bb18(%16, %20, %23, %25 : i16, i32, i64, i128)
  ^bb17(%27: i16, %28: i32, %29: i64, %30: i128, %31: i8):  // pred: ^bb1
    %32 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 0>"(%31) : (i8) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb19(%27, %28, %29, %30 : i16, i32, i64, i128)
  ^bb18(%33: i16, %34: i32, %35: i64, %36: i128):  // pred: ^bb16
    %37 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %38 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 1>"(%37) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb19(%33, %34, %35, %36 : i16, i32, i64, i128)
  ^bb19(%39: i16, %40: i32, %41: i64, %42: i128):  // 2 preds: ^bb17, ^bb18
    %43 = llvm.mlir.constant(256 : i16) : i16
    %44 = llvm.icmp "ult" %39, %43 : i16
    llvm.cond_br %44, ^bb2, ^bb21(%39, %40, %41, %42 : i16, i32, i64, i128)
  ^bb20(%45: i16, %46: i32, %47: i64, %48: i128, %49: i8):  // pred: ^bb2
    %50 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 0>"(%49) : (i8) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb22(%45, %46, %47, %48 : i16, i32, i64, i128)
  ^bb21(%51: i16, %52: i32, %53: i64, %54: i128):  // pred: ^bb19
    %55 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %56 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 1>"(%55) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb22(%51, %52, %53, %54 : i16, i32, i64, i128)
  ^bb22(%57: i16, %58: i32, %59: i64, %60: i128):  // 2 preds: ^bb20, ^bb21
    %61 = llvm.mlir.constant(true) : i1
    llvm.cond_br %61, ^bb3, ^bb24(%58, %59, %60 : i32, i64, i128)
  ^bb23(%62: i32, %63: i64, %64: i128, %65: i16):  // pred: ^bb3
    %66 = llvm.call @"enum_init<core::option::Option::<core::integer::u16>, 0>"(%65) : (i16) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb25(%62, %63, %64 : i32, i64, i128)
  ^bb24(%67: i32, %68: i64, %69: i128):  // pred: ^bb22
    %70 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %71 = llvm.call @"enum_init<core::option::Option::<core::integer::u16>, 1>"(%70) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb25(%67, %68, %69 : i32, i64, i128)
  ^bb25(%72: i32, %73: i64, %74: i128):  // 2 preds: ^bb23, ^bb24
    %75 = llvm.mlir.constant(256 : i32) : i32
    %76 = llvm.icmp "ult" %72, %75 : i32
    llvm.cond_br %76, ^bb4, ^bb27(%72, %73, %74 : i32, i64, i128)
  ^bb26(%77: i32, %78: i64, %79: i128, %80: i8):  // pred: ^bb4
    %81 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 0>"(%80) : (i8) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb28(%77, %78, %79 : i32, i64, i128)
  ^bb27(%82: i32, %83: i64, %84: i128):  // pred: ^bb25
    %85 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %86 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 1>"(%85) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb28(%82, %83, %84 : i32, i64, i128)
  ^bb28(%87: i32, %88: i64, %89: i128):  // 2 preds: ^bb26, ^bb27
    %90 = llvm.mlir.constant(65536 : i32) : i32
    %91 = llvm.icmp "ult" %87, %90 : i32
    llvm.cond_br %91, ^bb5, ^bb30(%87, %88, %89 : i32, i64, i128)
  ^bb29(%92: i32, %93: i64, %94: i128, %95: i16):  // pred: ^bb5
    %96 = llvm.call @"enum_init<core::option::Option::<core::integer::u16>, 0>"(%95) : (i16) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb31(%92, %93, %94 : i32, i64, i128)
  ^bb30(%97: i32, %98: i64, %99: i128):  // pred: ^bb28
    %100 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %101 = llvm.call @"enum_init<core::option::Option::<core::integer::u16>, 1>"(%100) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb31(%97, %98, %99 : i32, i64, i128)
  ^bb31(%102: i32, %103: i64, %104: i128):  // 2 preds: ^bb29, ^bb30
    %105 = llvm.mlir.constant(true) : i1
    llvm.cond_br %105, ^bb6, ^bb33(%103, %104 : i64, i128)
  ^bb32(%106: i64, %107: i128, %108: i32):  // pred: ^bb6
    %109 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%108) : (i32) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb34(%106, %107 : i64, i128)
  ^bb33(%110: i64, %111: i128):  // pred: ^bb31
    %112 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %113 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%112) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb34(%110, %111 : i64, i128)
  ^bb34(%114: i64, %115: i128):  // 2 preds: ^bb32, ^bb33
    %116 = llvm.mlir.constant(256 : i64) : i64
    %117 = llvm.icmp "ult" %114, %116 : i64
    llvm.cond_br %117, ^bb7, ^bb36(%114, %115 : i64, i128)
  ^bb35(%118: i64, %119: i128, %120: i8):  // pred: ^bb7
    %121 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 0>"(%120) : (i8) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb37(%118, %119 : i64, i128)
  ^bb36(%122: i64, %123: i128):  // pred: ^bb34
    %124 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %125 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 1>"(%124) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb37(%122, %123 : i64, i128)
  ^bb37(%126: i64, %127: i128):  // 2 preds: ^bb35, ^bb36
    %128 = llvm.mlir.constant(65536 : i64) : i64
    %129 = llvm.icmp "ult" %126, %128 : i64
    llvm.cond_br %129, ^bb8, ^bb39(%126, %127 : i64, i128)
  ^bb38(%130: i64, %131: i128, %132: i16):  // pred: ^bb8
    %133 = llvm.call @"enum_init<core::option::Option::<core::integer::u16>, 0>"(%132) : (i16) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb40(%130, %131 : i64, i128)
  ^bb39(%134: i64, %135: i128):  // pred: ^bb37
    %136 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %137 = llvm.call @"enum_init<core::option::Option::<core::integer::u16>, 1>"(%136) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb40(%134, %135 : i64, i128)
  ^bb40(%138: i64, %139: i128):  // 2 preds: ^bb38, ^bb39
    %140 = llvm.mlir.constant(4294967296 : i64) : i64
    %141 = llvm.icmp "ult" %138, %140 : i64
    llvm.cond_br %141, ^bb9, ^bb42(%138, %139 : i64, i128)
  ^bb41(%142: i64, %143: i128, %144: i32):  // pred: ^bb9
    %145 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%144) : (i32) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb43(%142, %143 : i64, i128)
  ^bb42(%146: i64, %147: i128):  // pred: ^bb40
    %148 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %149 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%148) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb43(%146, %147 : i64, i128)
  ^bb43(%150: i64, %151: i128):  // 2 preds: ^bb41, ^bb42
    %152 = llvm.mlir.constant(true) : i1
    llvm.cond_br %152, ^bb10, ^bb45(%151 : i128)
  ^bb44(%153: i128, %154: i64):  // pred: ^bb10
    %155 = llvm.call @"enum_init<core::option::Option::<core::integer::u64>, 0>"(%154) : (i64) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.br ^bb46(%153 : i128)
  ^bb45(%156: i128):  // pred: ^bb43
    %157 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %158 = llvm.call @"enum_init<core::option::Option::<core::integer::u64>, 1>"(%157) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.br ^bb46(%156 : i128)
  ^bb46(%159: i128):  // 2 preds: ^bb44, ^bb45
    %160 = llvm.mlir.constant(256 : i128) : i128
    %161 = llvm.icmp "ult" %159, %160 : i128
    llvm.cond_br %161, ^bb11, ^bb48(%159 : i128)
  ^bb47(%162: i128, %163: i8):  // pred: ^bb11
    %164 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 0>"(%163) : (i8) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb49(%162 : i128)
  ^bb48(%165: i128):  // pred: ^bb46
    %166 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %167 = llvm.call @"enum_init<core::option::Option::<core::integer::u8>, 1>"(%166) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<1 x i8>)>
    llvm.br ^bb49(%165 : i128)
  ^bb49(%168: i128):  // 2 preds: ^bb47, ^bb48
    %169 = llvm.mlir.constant(65536 : i128) : i128
    %170 = llvm.icmp "ult" %168, %169 : i128
    llvm.cond_br %170, ^bb12, ^bb51(%168 : i128)
  ^bb50(%171: i128, %172: i16):  // pred: ^bb12
    %173 = llvm.call @"enum_init<core::option::Option::<core::integer::u16>, 0>"(%172) : (i16) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb52(%171 : i128)
  ^bb51(%174: i128):  // pred: ^bb49
    %175 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %176 = llvm.call @"enum_init<core::option::Option::<core::integer::u16>, 1>"(%175) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb52(%174 : i128)
  ^bb52(%177: i128):  // 2 preds: ^bb50, ^bb51
    %178 = llvm.mlir.constant(4294967296 : i128) : i128
    %179 = llvm.icmp "ult" %177, %178 : i128
    llvm.cond_br %179, ^bb13, ^bb54(%177 : i128)
  ^bb53(%180: i128, %181: i32):  // pred: ^bb13
    %182 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%181) : (i32) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb55(%180 : i128)
  ^bb54(%183: i128):  // pred: ^bb52
    %184 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %185 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%184) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb55(%183 : i128)
  ^bb55(%186: i128):  // 2 preds: ^bb53, ^bb54
    %187 = llvm.mlir.constant(18446744073709551616 : i128) : i128
    %188 = llvm.icmp "ult" %186, %187 : i128
    llvm.cond_br %188, ^bb14, ^bb57(%186 : i128)
  ^bb56(%189: i128, %190: i64):  // pred: ^bb14
    %191 = llvm.call @"enum_init<core::option::Option::<core::integer::u64>, 0>"(%190) : (i64) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.br ^bb58(%189 : i128)
  ^bb57(%192: i128):  // pred: ^bb55
    %193 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %194 = llvm.call @"enum_init<core::option::Option::<core::integer::u64>, 1>"(%193) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.br ^bb58(%192 : i128)
  ^bb58(%195: i128):  // 2 preds: ^bb56, ^bb57
    %196 = llvm.mlir.constant(true) : i1
    llvm.cond_br %196, ^bb15, ^bb60
  ^bb59(%197: i128):  // pred: ^bb15
    %198 = llvm.call @"enum_init<core::option::Option::<core::integer::u128>, 0>"(%197) : (i128) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.br ^bb61
  ^bb60:  // pred: ^bb58
    %199 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %200 = llvm.call @"enum_init<core::option::Option::<core::integer::u128>, 1>"(%199) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.br ^bb61
  ^bb61:  // 2 preds: ^bb59, ^bb60
    %201 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    llvm.return %201 : !llvm.struct<packed ()>
  }
  llvm.func @"_mlir_ciface_casts::casts::main"(%arg0: !llvm.ptr<struct<packed ()>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"casts::casts::main"() : () -> !llvm.struct<packed ()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed ()>>
    llvm.return
  }
}
