module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"struct_deconstruct<Tuple<u16>>"(%arg0: !llvm.struct<packed (i16)>) -> i16 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i16)> 
    llvm.return %0 : i16
  }
  llvm.func internal @"struct_deconstruct<Tuple<u32>>"(%arg0: !llvm.struct<packed (i32)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_deconstruct<Tuple<u64>>"(%arg0: !llvm.struct<packed (i64)>) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i64)> 
    llvm.return %0 : i64
  }
  llvm.func internal @"struct_deconstruct<Tuple<u128>>"(%arg0: !llvm.struct<packed (i128)>) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i128)> 
    llvm.return %0 : i128
  }
  llvm.func internal @"struct_construct<Tuple<u16, u16, u16>>"(%arg0: i16, %arg1: i16, %arg2: i16) -> !llvm.struct<packed (i16, i16, i16)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i16, i16, i16)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i16, i16, i16)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<packed (i16, i16, i16)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<packed (i16, i16, i16)> 
    llvm.return %3 : !llvm.struct<packed (i16, i16, i16)>
  }
  llvm.func internal @"struct_construct<Tuple<u32, u32, u32>>"(%arg0: i32, %arg1: i32, %arg2: i32) -> !llvm.struct<packed (i32, i32, i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32, i32, i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i32, i32, i32)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<packed (i32, i32, i32)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<packed (i32, i32, i32)> 
    llvm.return %3 : !llvm.struct<packed (i32, i32, i32)>
  }
  llvm.func internal @"struct_construct<Tuple<u64, u64, u64>>"(%arg0: i64, %arg1: i64, %arg2: i64) -> !llvm.struct<packed (i64, i64, i64)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i64, i64, i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i64, i64, i64)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<packed (i64, i64, i64)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<packed (i64, i64, i64)> 
    llvm.return %3 : !llvm.struct<packed (i64, i64, i64)>
  }
  llvm.func internal @"struct_construct<Tuple<u128, u128, u128>>"(%arg0: i128, %arg1: i128, %arg2: i128) -> !llvm.struct<packed (i128, i128, i128)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i128, i128, i128)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i128, i128, i128)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<packed (i128, i128, i128)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<packed (i128, i128, i128)> 
    llvm.return %3 : !llvm.struct<packed (i128, i128, i128)>
  }
  llvm.func internal @"struct_construct<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>"(%arg0: !llvm.struct<packed (i16, i16, i16)>, %arg1: !llvm.struct<packed (i32, i32, i32)>, %arg2: !llvm.struct<packed (i64, i64, i64)>, %arg3: !llvm.struct<packed (i128, i128, i128)>) -> !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)> 
    %4 = llvm.insertvalue %arg3, %3[3] : !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)> 
    llvm.return %4 : !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>
  }
  llvm.func internal @"struct_construct<Tuple<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>>"(%arg0: !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>) -> !llvm.struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)> 
    llvm.return %1 : !llvm.struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 0>"(%arg0: !llvm.struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)>) -> !llvm.struct<packed (i16, array<90 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<90 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<90 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 0>"(%arg0: i16) -> !llvm.struct<packed (i16, array<2 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
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
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 1>"(%arg0: i16) -> !llvm.struct<packed (i16, array<2 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<2 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i16)>
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<2 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<u16>>"(%arg0: i16) -> !llvm.struct<packed (i16)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i16)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i16)> 
    llvm.return %1 : !llvm.struct<packed (i16)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(%arg0: !llvm.struct<packed (i16)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i16)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i16)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 0>"(%arg0: i32) -> !llvm.struct<packed (i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
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
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 1>"(%arg0: i32) -> !llvm.struct<packed (i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i32)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<4 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<u32>>"(%arg0: i32) -> !llvm.struct<packed (i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i32)> 
    llvm.return %1 : !llvm.struct<packed (i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(%arg0: !llvm.struct<packed (i32)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i32)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 0>"(%arg0: i64) -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
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
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 1>"(%arg0: i64) -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i64)>
    llvm.store %arg0, %4 : i64, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<u64>>"(%arg0: i64) -> !llvm.struct<packed (i64)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i64)> 
    llvm.return %1 : !llvm.struct<packed (i64)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(%arg0: !llvm.struct<packed (i64)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i64)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i64)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 0>"(%arg0: i128) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
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
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 1>"(%arg0: i128) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i128)>
    llvm.store %arg0, %4 : i128, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<u128>>"(%arg0: i128) -> !llvm.struct<packed (i128)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i128)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i128)> 
    llvm.return %1 : !llvm.struct<packed (i128)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(%arg0: !llvm.struct<packed (i128)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i128)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i128)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"array_new<felt252>"() -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"array_append<felt252>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i256) -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<packed (i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<packed (i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    llvm.store %arg1, %12 : i256, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func @"uint_addition::uint_addition::main"() -> !llvm.struct<packed (i16, array<90 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb37
  ^bb1:  // pred: ^bb37
    %0 = llvm.getelementptr inbounds %52[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i16)>)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<packed (i16)>
    llvm.br ^bb38(%1 : !llvm.struct<packed (i16)>)
  ^bb2:  // pred: ^bb37
    %2 = llvm.getelementptr inbounds %52[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb61(%3 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb3:  // pred: ^bb37
    llvm.unreachable
  ^bb4:  // pred: ^bb38
    %4 = llvm.getelementptr inbounds %61[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i16)>)>
    %5 = llvm.load %4 : !llvm.ptr -> !llvm.struct<packed (i16)>
    llvm.br ^bb39(%56, %5 : i16, !llvm.struct<packed (i16)>)
  ^bb5:  // pred: ^bb38
    %6 = llvm.getelementptr inbounds %61[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %7 = llvm.load %6 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb60(%7 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb6:  // pred: ^bb38
    llvm.unreachable
  ^bb7:  // pred: ^bb39
    %8 = llvm.getelementptr inbounds %71[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i16)>)>
    %9 = llvm.load %8 : !llvm.ptr -> !llvm.struct<packed (i16)>
    llvm.br ^bb40(%64, %66, %9 : i16, i16, !llvm.struct<packed (i16)>)
  ^bb8:  // pred: ^bb39
    %10 = llvm.getelementptr inbounds %71[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %11 = llvm.load %10 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb59(%11 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb9:  // pred: ^bb39
    llvm.unreachable
  ^bb10:  // pred: ^bb40
    %12 = llvm.getelementptr inbounds %82[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %13 = llvm.load %12 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb41(%74, %75, %77, %13 : i16, i16, i16, !llvm.struct<packed (i32)>)
  ^bb11:  // pred: ^bb40
    %14 = llvm.getelementptr inbounds %82[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %15 = llvm.load %14 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb58(%15 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb12:  // pred: ^bb40
    llvm.unreachable
  ^bb13:  // pred: ^bb41
    %16 = llvm.getelementptr inbounds %94[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb42(%85, %86, %87, %89, %17 : i16, i16, i16, i32, !llvm.struct<packed (i32)>)
  ^bb14:  // pred: ^bb41
    %18 = llvm.getelementptr inbounds %94[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %19 = llvm.load %18 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb57(%19 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb15:  // pred: ^bb41
    llvm.unreachable
  ^bb16:  // pred: ^bb42
    %20 = llvm.getelementptr inbounds %107[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %21 = llvm.load %20 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb43(%97, %98, %99, %100, %102, %21 : i16, i16, i16, i32, i32, !llvm.struct<packed (i32)>)
  ^bb17:  // pred: ^bb42
    %22 = llvm.getelementptr inbounds %107[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %23 = llvm.load %22 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb56(%23 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb18:  // pred: ^bb42
    llvm.unreachable
  ^bb19:  // pred: ^bb43
    %24 = llvm.getelementptr inbounds %121[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i64)>)>
    %25 = llvm.load %24 : !llvm.ptr -> !llvm.struct<packed (i64)>
    llvm.br ^bb44(%110, %111, %112, %113, %114, %116, %25 : i16, i16, i16, i32, i32, i32, !llvm.struct<packed (i64)>)
  ^bb20:  // pred: ^bb43
    %26 = llvm.getelementptr inbounds %121[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %27 = llvm.load %26 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb55(%27 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb21:  // pred: ^bb43
    llvm.unreachable
  ^bb22:  // pred: ^bb44
    %28 = llvm.getelementptr inbounds %136[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i64)>)>
    %29 = llvm.load %28 : !llvm.ptr -> !llvm.struct<packed (i64)>
    llvm.br ^bb45(%124, %125, %126, %127, %128, %129, %131, %29 : i16, i16, i16, i32, i32, i32, i64, !llvm.struct<packed (i64)>)
  ^bb23:  // pred: ^bb44
    %30 = llvm.getelementptr inbounds %136[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %31 = llvm.load %30 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb54(%31 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb24:  // pred: ^bb44
    llvm.unreachable
  ^bb25:  // pred: ^bb45
    %32 = llvm.getelementptr inbounds %152[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i64)>)>
    %33 = llvm.load %32 : !llvm.ptr -> !llvm.struct<packed (i64)>
    llvm.br ^bb46(%139, %140, %141, %142, %143, %144, %145, %147, %33 : i16, i16, i16, i32, i32, i32, i64, i64, !llvm.struct<packed (i64)>)
  ^bb26:  // pred: ^bb45
    %34 = llvm.getelementptr inbounds %152[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %35 = llvm.load %34 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb53(%35 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb27:  // pred: ^bb45
    llvm.unreachable
  ^bb28:  // pred: ^bb46
    %36 = llvm.getelementptr inbounds %169[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i128)>)>
    %37 = llvm.load %36 : !llvm.ptr -> !llvm.struct<packed (i128)>
    llvm.br ^bb47(%155, %156, %157, %158, %159, %160, %161, %162, %164, %37 : i16, i16, i16, i32, i32, i32, i64, i64, i64, !llvm.struct<packed (i128)>)
  ^bb29:  // pred: ^bb46
    %38 = llvm.getelementptr inbounds %169[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %39 = llvm.load %38 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb52(%39 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb30:  // pred: ^bb46
    llvm.unreachable
  ^bb31:  // pred: ^bb47
    %40 = llvm.getelementptr inbounds %187[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i128)>)>
    %41 = llvm.load %40 : !llvm.ptr -> !llvm.struct<packed (i128)>
    llvm.br ^bb48(%172, %173, %174, %175, %176, %177, %178, %179, %180, %182, %41 : i16, i16, i16, i32, i32, i32, i64, i64, i64, i128, !llvm.struct<packed (i128)>)
  ^bb32:  // pred: ^bb47
    %42 = llvm.getelementptr inbounds %187[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %43 = llvm.load %42 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb51(%43 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb33:  // pred: ^bb47
    llvm.unreachable
  ^bb34:  // pred: ^bb48
    %44 = llvm.getelementptr inbounds %206[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i128)>)>
    %45 = llvm.load %44 : !llvm.ptr -> !llvm.struct<packed (i128)>
    llvm.br ^bb49(%190, %191, %192, %193, %194, %195, %196, %197, %198, %199, %201, %45 : i16, i16, i16, i32, i32, i32, i64, i64, i64, i128, i128, !llvm.struct<packed (i128)>)
  ^bb35:  // pred: ^bb48
    %46 = llvm.getelementptr inbounds %206[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %47 = llvm.load %46 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb50(%47 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb36:  // pred: ^bb48
    llvm.unreachable
  ^bb37:  // pred: ^bb0
    %48 = llvm.mlir.constant(4 : i16) : i16
    %49 = llvm.mlir.constant(6 : i16) : i16
    %50 = llvm.call @"core::integer::U16Add::add"(%48, %49) : (i16, i16) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %51 = llvm.mlir.constant(1 : i64) : i64
    %52 = llvm.alloca %51 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %50, %52 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %53 = llvm.getelementptr inbounds %52[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %54 = llvm.load %53 : !llvm.ptr -> i16
    llvm.switch %54 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb38(%55: !llvm.struct<packed (i16)>):  // pred: ^bb1
    %56 = llvm.call @"struct_deconstruct<Tuple<u16>>"(%55) : (!llvm.struct<packed (i16)>) -> i16
    %57 = llvm.mlir.constant(2 : i16) : i16
    %58 = llvm.mlir.constant(10 : i16) : i16
    %59 = llvm.call @"core::integer::U16Add::add"(%57, %58) : (i16, i16) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %60 = llvm.mlir.constant(1 : i64) : i64
    %61 = llvm.alloca %60 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %59, %61 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %62 = llvm.getelementptr inbounds %61[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %63 = llvm.load %62 : !llvm.ptr -> i16
    llvm.switch %63 : i16, ^bb6 [
      0: ^bb4,
      1: ^bb5
    ]
  ^bb39(%64: i16, %65: !llvm.struct<packed (i16)>):  // pred: ^bb4
    %66 = llvm.call @"struct_deconstruct<Tuple<u16>>"(%65) : (!llvm.struct<packed (i16)>) -> i16
    %67 = llvm.mlir.constant(50 : i16) : i16
    %68 = llvm.mlir.constant(2 : i16) : i16
    %69 = llvm.call @"core::integer::U16Add::add"(%67, %68) : (i16, i16) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %70 = llvm.mlir.constant(1 : i64) : i64
    %71 = llvm.alloca %70 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %69, %71 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %72 = llvm.getelementptr inbounds %71[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %73 = llvm.load %72 : !llvm.ptr -> i16
    llvm.switch %73 : i16, ^bb9 [
      0: ^bb7,
      1: ^bb8
    ]
  ^bb40(%74: i16, %75: i16, %76: !llvm.struct<packed (i16)>):  // pred: ^bb7
    %77 = llvm.call @"struct_deconstruct<Tuple<u16>>"(%76) : (!llvm.struct<packed (i16)>) -> i16
    %78 = llvm.mlir.constant(4 : i32) : i32
    %79 = llvm.mlir.constant(6 : i32) : i32
    %80 = llvm.call @"core::integer::U32Add::add"(%78, %79) : (i32, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %81 = llvm.mlir.constant(1 : i64) : i64
    %82 = llvm.alloca %81 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %80, %82 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %83 = llvm.getelementptr inbounds %82[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %84 = llvm.load %83 : !llvm.ptr -> i16
    llvm.switch %84 : i16, ^bb12 [
      0: ^bb10,
      1: ^bb11
    ]
  ^bb41(%85: i16, %86: i16, %87: i16, %88: !llvm.struct<packed (i32)>):  // pred: ^bb10
    %89 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%88) : (!llvm.struct<packed (i32)>) -> i32
    %90 = llvm.mlir.constant(2 : i32) : i32
    %91 = llvm.mlir.constant(10 : i32) : i32
    %92 = llvm.call @"core::integer::U32Add::add"(%90, %91) : (i32, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %93 = llvm.mlir.constant(1 : i64) : i64
    %94 = llvm.alloca %93 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %92, %94 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %95 = llvm.getelementptr inbounds %94[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %96 = llvm.load %95 : !llvm.ptr -> i16
    llvm.switch %96 : i16, ^bb15 [
      0: ^bb13,
      1: ^bb14
    ]
  ^bb42(%97: i16, %98: i16, %99: i16, %100: i32, %101: !llvm.struct<packed (i32)>):  // pred: ^bb13
    %102 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%101) : (!llvm.struct<packed (i32)>) -> i32
    %103 = llvm.mlir.constant(50 : i32) : i32
    %104 = llvm.mlir.constant(2 : i32) : i32
    %105 = llvm.call @"core::integer::U32Add::add"(%103, %104) : (i32, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %106 = llvm.mlir.constant(1 : i64) : i64
    %107 = llvm.alloca %106 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %105, %107 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %108 = llvm.getelementptr inbounds %107[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %109 = llvm.load %108 : !llvm.ptr -> i16
    llvm.switch %109 : i16, ^bb18 [
      0: ^bb16,
      1: ^bb17
    ]
  ^bb43(%110: i16, %111: i16, %112: i16, %113: i32, %114: i32, %115: !llvm.struct<packed (i32)>):  // pred: ^bb16
    %116 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%115) : (!llvm.struct<packed (i32)>) -> i32
    %117 = llvm.mlir.constant(4 : i64) : i64
    %118 = llvm.mlir.constant(6 : i64) : i64
    %119 = llvm.call @"core::integer::U64Add::add"(%117, %118) : (i64, i64) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %120 = llvm.mlir.constant(1 : i64) : i64
    %121 = llvm.alloca %120 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %119, %121 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %122 = llvm.getelementptr inbounds %121[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %123 = llvm.load %122 : !llvm.ptr -> i16
    llvm.switch %123 : i16, ^bb21 [
      0: ^bb19,
      1: ^bb20
    ]
  ^bb44(%124: i16, %125: i16, %126: i16, %127: i32, %128: i32, %129: i32, %130: !llvm.struct<packed (i64)>):  // pred: ^bb19
    %131 = llvm.call @"struct_deconstruct<Tuple<u64>>"(%130) : (!llvm.struct<packed (i64)>) -> i64
    %132 = llvm.mlir.constant(2 : i64) : i64
    %133 = llvm.mlir.constant(10 : i64) : i64
    %134 = llvm.call @"core::integer::U64Add::add"(%132, %133) : (i64, i64) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %135 = llvm.mlir.constant(1 : i64) : i64
    %136 = llvm.alloca %135 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %134, %136 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %137 = llvm.getelementptr inbounds %136[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %138 = llvm.load %137 : !llvm.ptr -> i16
    llvm.switch %138 : i16, ^bb24 [
      0: ^bb22,
      1: ^bb23
    ]
  ^bb45(%139: i16, %140: i16, %141: i16, %142: i32, %143: i32, %144: i32, %145: i64, %146: !llvm.struct<packed (i64)>):  // pred: ^bb22
    %147 = llvm.call @"struct_deconstruct<Tuple<u64>>"(%146) : (!llvm.struct<packed (i64)>) -> i64
    %148 = llvm.mlir.constant(50 : i64) : i64
    %149 = llvm.mlir.constant(2 : i64) : i64
    %150 = llvm.call @"core::integer::U64Add::add"(%148, %149) : (i64, i64) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %151 = llvm.mlir.constant(1 : i64) : i64
    %152 = llvm.alloca %151 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %150, %152 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %153 = llvm.getelementptr inbounds %152[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %154 = llvm.load %153 : !llvm.ptr -> i16
    llvm.switch %154 : i16, ^bb27 [
      0: ^bb25,
      1: ^bb26
    ]
  ^bb46(%155: i16, %156: i16, %157: i16, %158: i32, %159: i32, %160: i32, %161: i64, %162: i64, %163: !llvm.struct<packed (i64)>):  // pred: ^bb25
    %164 = llvm.call @"struct_deconstruct<Tuple<u64>>"(%163) : (!llvm.struct<packed (i64)>) -> i64
    %165 = llvm.mlir.constant(4 : i128) : i128
    %166 = llvm.mlir.constant(6 : i128) : i128
    %167 = llvm.call @"core::integer::U128Add::add"(%165, %166) : (i128, i128) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %168 = llvm.mlir.constant(1 : i64) : i64
    %169 = llvm.alloca %168 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %167, %169 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %170 = llvm.getelementptr inbounds %169[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %171 = llvm.load %170 : !llvm.ptr -> i16
    llvm.switch %171 : i16, ^bb30 [
      0: ^bb28,
      1: ^bb29
    ]
  ^bb47(%172: i16, %173: i16, %174: i16, %175: i32, %176: i32, %177: i32, %178: i64, %179: i64, %180: i64, %181: !llvm.struct<packed (i128)>):  // pred: ^bb28
    %182 = llvm.call @"struct_deconstruct<Tuple<u128>>"(%181) : (!llvm.struct<packed (i128)>) -> i128
    %183 = llvm.mlir.constant(2 : i128) : i128
    %184 = llvm.mlir.constant(10 : i128) : i128
    %185 = llvm.call @"core::integer::U128Add::add"(%183, %184) : (i128, i128) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %186 = llvm.mlir.constant(1 : i64) : i64
    %187 = llvm.alloca %186 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %185, %187 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %188 = llvm.getelementptr inbounds %187[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %189 = llvm.load %188 : !llvm.ptr -> i16
    llvm.switch %189 : i16, ^bb33 [
      0: ^bb31,
      1: ^bb32
    ]
  ^bb48(%190: i16, %191: i16, %192: i16, %193: i32, %194: i32, %195: i32, %196: i64, %197: i64, %198: i64, %199: i128, %200: !llvm.struct<packed (i128)>):  // pred: ^bb31
    %201 = llvm.call @"struct_deconstruct<Tuple<u128>>"(%200) : (!llvm.struct<packed (i128)>) -> i128
    %202 = llvm.mlir.constant(50 : i128) : i128
    %203 = llvm.mlir.constant(2 : i128) : i128
    %204 = llvm.call @"core::integer::U128Add::add"(%202, %203) : (i128, i128) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %205 = llvm.mlir.constant(1 : i64) : i64
    %206 = llvm.alloca %205 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %204, %206 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %207 = llvm.getelementptr inbounds %206[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %208 = llvm.load %207 : !llvm.ptr -> i16
    llvm.switch %208 : i16, ^bb36 [
      0: ^bb34,
      1: ^bb35
    ]
  ^bb49(%209: i16, %210: i16, %211: i16, %212: i32, %213: i32, %214: i32, %215: i64, %216: i64, %217: i64, %218: i128, %219: i128, %220: !llvm.struct<packed (i128)>):  // pred: ^bb34
    %221 = llvm.call @"struct_deconstruct<Tuple<u128>>"(%220) : (!llvm.struct<packed (i128)>) -> i128
    %222 = llvm.call @"struct_construct<Tuple<u16, u16, u16>>"(%209, %210, %211) : (i16, i16, i16) -> !llvm.struct<packed (i16, i16, i16)>
    %223 = llvm.call @"struct_construct<Tuple<u32, u32, u32>>"(%212, %213, %214) : (i32, i32, i32) -> !llvm.struct<packed (i32, i32, i32)>
    %224 = llvm.call @"struct_construct<Tuple<u64, u64, u64>>"(%215, %216, %217) : (i64, i64, i64) -> !llvm.struct<packed (i64, i64, i64)>
    %225 = llvm.call @"struct_construct<Tuple<u128, u128, u128>>"(%218, %219, %221) : (i128, i128, i128) -> !llvm.struct<packed (i128, i128, i128)>
    %226 = llvm.call @"struct_construct<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>"(%222, %223, %224, %225) : (!llvm.struct<packed (i16, i16, i16)>, !llvm.struct<packed (i32, i32, i32)>, !llvm.struct<packed (i64, i64, i64)>, !llvm.struct<packed (i128, i128, i128)>) -> !llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>
    %227 = llvm.call @"struct_construct<Tuple<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>>"(%226) : (!llvm.struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>) -> !llvm.struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)>
    %228 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 0>"(%227) : (!llvm.struct<packed (struct<packed (struct<packed (i16, i16, i16)>, struct<packed (i32, i32, i32)>, struct<packed (i64, i64, i64)>, struct<packed (i128, i128, i128)>)>)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %228 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb50(%229: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb35
    %230 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%229) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %230 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb51(%231: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb32
    %232 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%231) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %232 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb52(%233: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb29
    %234 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%233) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %234 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb53(%235: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb26
    %236 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%235) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %236 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb54(%237: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb23
    %238 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%237) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %238 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb55(%239: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb20
    %240 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%239) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %240 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb56(%241: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb17
    %242 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%241) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %242 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb57(%243: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb14
    %244 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%243) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %244 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb58(%245: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb11
    %246 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%245) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %246 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb59(%247: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb8
    %248 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%247) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %248 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb60(%249: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb5
    %250 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%249) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %250 : !llvm.struct<packed (i16, array<90 x i8>)>
  ^bb61(%251: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb2
    %252 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%251) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.return %252 : !llvm.struct<packed (i16, array<90 x i8>)>
  }
  llvm.func @"_mlir_ciface_uint_addition::uint_addition::main"(%arg0: !llvm.ptr<struct<packed (i16, array<90 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"uint_addition::uint_addition::main"() : () -> !llvm.struct<packed (i16, array<90 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<90 x i8>)>>
    llvm.return
  }
  llvm.func @"core::integer::U16Add::add"(%arg0: i16, %arg1: i16) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : i16, i16)
  ^bb1:  // pred: ^bb7
    %0 = llvm.getelementptr inbounds %17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i16)>)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<packed (i16)>
    llvm.br ^bb8(%1 : !llvm.struct<packed (i16)>)
  ^bb2:  // pred: ^bb7
    %2 = llvm.getelementptr inbounds %17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb9(%3 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb3:  // pred: ^bb7
    llvm.unreachable
  ^bb4(%4: i16, %5: i16):  // pred: ^bb0
    %6 = "llvm.intr.uadd.with.overflow"(%4, %5) : (i16, i16) -> !llvm.struct<packed (i16, i1)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<packed (i16, i1)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<packed (i16, i1)> 
    llvm.cond_br %8, ^bb6(%7 : i16), ^bb5(%7 : i16)
  ^bb5(%9: i16):  // pred: ^bb4
    %10 = llvm.call @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 0>"(%9) : (i16) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb7(%10 : !llvm.struct<packed (i16, array<2 x i8>)>)
  ^bb6(%11: i16):  // pred: ^bb4
    %12 = llvm.call @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 1>"(%11) : (i16) -> !llvm.struct<packed (i16, array<2 x i8>)>
    llvm.br ^bb7(%12 : !llvm.struct<packed (i16, array<2 x i8>)>)
  ^bb7(%13: !llvm.struct<packed (i16, array<2 x i8>)>):  // 2 preds: ^bb5, ^bb6
    %14 = llvm.mlir.constant(155775200859838811096160292336445452151 : i256) : i256
    %15 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(%13, %14) : (!llvm.struct<packed (i16, array<2 x i8>)>, i256) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %19 = llvm.load %18 : !llvm.ptr -> i16
    llvm.switch %19 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb8(%20: !llvm.struct<packed (i16)>):  // pred: ^bb1
    %21 = llvm.call @"struct_deconstruct<Tuple<u16>>"(%20) : (!llvm.struct<packed (i16)>) -> i16
    %22 = llvm.call @"struct_construct<Tuple<u16>>"(%21) : (i16) -> !llvm.struct<packed (i16)>
    %23 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(%22) : (!llvm.struct<packed (i16)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %23 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb9(%24: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb2
    %25 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(%24) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %25 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U16Add::add"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: i16, %arg2: i16) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::integer::U16Add::add"(%arg1, %arg2) : (i16, i16) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::integer::U32Add::add"(%arg0: i32, %arg1: i32) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : i32, i32)
  ^bb1:  // pred: ^bb7
    %0 = llvm.getelementptr inbounds %17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb8(%1 : !llvm.struct<packed (i32)>)
  ^bb2:  // pred: ^bb7
    %2 = llvm.getelementptr inbounds %17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb9(%3 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb3:  // pred: ^bb7
    llvm.unreachable
  ^bb4(%4: i32, %5: i32):  // pred: ^bb0
    %6 = "llvm.intr.uadd.with.overflow"(%4, %5) : (i32, i32) -> !llvm.struct<packed (i32, i1)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<packed (i32, i1)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<packed (i32, i1)> 
    llvm.cond_br %8, ^bb6(%7 : i32), ^bb5(%7 : i32)
  ^bb5(%9: i32):  // pred: ^bb4
    %10 = llvm.call @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 0>"(%9) : (i32) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb7(%10 : !llvm.struct<packed (i16, array<4 x i8>)>)
  ^bb6(%11: i32):  // pred: ^bb4
    %12 = llvm.call @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 1>"(%11) : (i32) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb7(%12 : !llvm.struct<packed (i16, array<4 x i8>)>)
  ^bb7(%13: !llvm.struct<packed (i16, array<4 x i8>)>):  // 2 preds: ^bb5, ^bb6
    %14 = llvm.mlir.constant(155785504323917466144735657540098748279 : i256) : i256
    %15 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(%13, %14) : (!llvm.struct<packed (i16, array<4 x i8>)>, i256) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %19 = llvm.load %18 : !llvm.ptr -> i16
    llvm.switch %19 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb8(%20: !llvm.struct<packed (i32)>):  // pred: ^bb1
    %21 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%20) : (!llvm.struct<packed (i32)>) -> i32
    %22 = llvm.call @"struct_construct<Tuple<u32>>"(%21) : (i32) -> !llvm.struct<packed (i32)>
    %23 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(%22) : (!llvm.struct<packed (i32)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %23 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb9(%24: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb2
    %25 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(%24) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %25 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U32Add::add"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: i32, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::integer::U32Add::add"(%arg1, %arg2) : (i32, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::integer::U64Add::add"(%arg0: i64, %arg1: i64) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : i64, i64)
  ^bb1:  // pred: ^bb7
    %0 = llvm.getelementptr inbounds %17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i64)>)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<packed (i64)>
    llvm.br ^bb8(%1 : !llvm.struct<packed (i64)>)
  ^bb2:  // pred: ^bb7
    %2 = llvm.getelementptr inbounds %17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb9(%3 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb3:  // pred: ^bb7
    llvm.unreachable
  ^bb4(%4: i64, %5: i64):  // pred: ^bb0
    %6 = "llvm.intr.uadd.with.overflow"(%4, %5) : (i64, i64) -> !llvm.struct<packed (i64, i1)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<packed (i64, i1)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<packed (i64, i1)> 
    llvm.cond_br %8, ^bb6(%7 : i64), ^bb5(%7 : i64)
  ^bb5(%9: i64):  // pred: ^bb4
    %10 = llvm.call @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 0>"(%9) : (i64) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.br ^bb7(%10 : !llvm.struct<packed (i16, array<8 x i8>)>)
  ^bb6(%11: i64):  // pred: ^bb4
    %12 = llvm.call @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 1>"(%11) : (i64) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.br ^bb7(%12 : !llvm.struct<packed (i16, array<8 x i8>)>)
  ^bb7(%13: !llvm.struct<packed (i16, array<8 x i8>)>):  // 2 preds: ^bb5, ^bb6
    %14 = llvm.mlir.constant(155801121779312277930962096923588980599 : i256) : i256
    %15 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(%13, %14) : (!llvm.struct<packed (i16, array<8 x i8>)>, i256) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %19 = llvm.load %18 : !llvm.ptr -> i16
    llvm.switch %19 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb8(%20: !llvm.struct<packed (i64)>):  // pred: ^bb1
    %21 = llvm.call @"struct_deconstruct<Tuple<u64>>"(%20) : (!llvm.struct<packed (i64)>) -> i64
    %22 = llvm.call @"struct_construct<Tuple<u64>>"(%21) : (i64) -> !llvm.struct<packed (i64)>
    %23 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(%22) : (!llvm.struct<packed (i64)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %23 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb9(%24: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb2
    %25 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(%24) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %25 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U64Add::add"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: i64, %arg2: i64) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::integer::U64Add::add"(%arg1, %arg2) : (i64, i64) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::integer::U128Add::add"(%arg0: i128, %arg1: i128) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : i128, i128)
  ^bb1:  // pred: ^bb7
    %0 = llvm.getelementptr inbounds %17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i128)>)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<packed (i128)>
    llvm.br ^bb8(%1 : !llvm.struct<packed (i128)>)
  ^bb2:  // pred: ^bb7
    %2 = llvm.getelementptr inbounds %17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb9(%3 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb3:  // pred: ^bb7
    llvm.unreachable
  ^bb4(%4: i128, %5: i128):  // pred: ^bb0
    %6 = "llvm.intr.uadd.with.overflow"(%4, %5) : (i128, i128) -> !llvm.struct<packed (i128, i1)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<packed (i128, i1)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<packed (i128, i1)> 
    llvm.cond_br %8, ^bb6(%7 : i128), ^bb5(%7 : i128)
  ^bb5(%9: i128):  // pred: ^bb4
    %10 = llvm.call @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 0>"(%9) : (i128) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.br ^bb7(%10 : !llvm.struct<packed (i16, array<16 x i8>)>)
  ^bb6(%11: i128):  // pred: ^bb4
    %12 = llvm.call @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 1>"(%11) : (i128) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.br ^bb7(%12 : !llvm.struct<packed (i16, array<16 x i8>)>)
  ^bb7(%13: !llvm.struct<packed (i16, array<16 x i8>)>):  // 2 preds: ^bb5, ^bb6
    %14 = llvm.mlir.constant(39878429859757942499084499860145094553463 : i256) : i256
    %15 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(%13, %14) : (!llvm.struct<packed (i16, array<16 x i8>)>, i256) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %19 = llvm.load %18 : !llvm.ptr -> i16
    llvm.switch %19 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb8(%20: !llvm.struct<packed (i128)>):  // pred: ^bb1
    %21 = llvm.call @"struct_deconstruct<Tuple<u128>>"(%20) : (!llvm.struct<packed (i128)>) -> i128
    %22 = llvm.call @"struct_construct<Tuple<u128>>"(%21) : (i128) -> !llvm.struct<packed (i128)>
    %23 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(%22) : (!llvm.struct<packed (i128)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %23 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb9(%24: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb2
    %25 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(%24) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %25 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U128Add::add"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: i128, %arg2: i128) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::integer::U128Add::add"(%arg1, %arg2) : (i128, i128) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(%arg0: !llvm.struct<packed (i16, array<2 x i8>)>, %arg1: i256) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<packed (i16, array<2 x i8>)>, i256)
  ^bb1:  // pred: ^bb4
    %0 = llvm.getelementptr inbounds %5[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i16)>
    %1 = llvm.load %0 : !llvm.ptr -> i16
    llvm.br ^bb5(%1 : i16)
  ^bb2:  // pred: ^bb4
    llvm.br ^bb6(%3 : i256)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%2: !llvm.struct<packed (i16, array<2 x i8>)>, %3: i256):  // pred: ^bb0
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.struct<packed (i16, array<2 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %2, %5 : !llvm.struct<packed (i16, array<2 x i8>)>, !llvm.ptr
    %6 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<2 x i8>)>
    %7 = llvm.load %6 : !llvm.ptr -> i16
    llvm.switch %7 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%8: i16):  // pred: ^bb1
    %9 = llvm.call @"struct_construct<Tuple<u16>>"(%8) : (i16) -> !llvm.struct<packed (i16)>
    %10 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(%9) : (!llvm.struct<packed (i16)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %10 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb6(%11: i256):  // pred: ^bb2
    %12 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<packed (i32, i32, ptr)>
    %13 = llvm.call @"array_append<felt252>"(%12, %11) : (!llvm.struct<packed (i32, i32, ptr)>, i256) -> !llvm.struct<packed (i32, i32, ptr)>
    %14 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(%13) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %14 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: !llvm.struct<packed (i16, array<2 x i8>)>, %arg2: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(%arg1, %arg2) : (!llvm.struct<packed (i16, array<2 x i8>)>, i256) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(%arg0: !llvm.struct<packed (i16, array<4 x i8>)>, %arg1: i256) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<packed (i16, array<4 x i8>)>, i256)
  ^bb1:  // pred: ^bb4
    %0 = llvm.getelementptr inbounds %5[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i32)>
    %1 = llvm.load %0 : !llvm.ptr -> i32
    llvm.br ^bb5(%1 : i32)
  ^bb2:  // pred: ^bb4
    llvm.br ^bb6(%3 : i256)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%2: !llvm.struct<packed (i16, array<4 x i8>)>, %3: i256):  // pred: ^bb0
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.struct<packed (i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %2, %5 : !llvm.struct<packed (i16, array<4 x i8>)>, !llvm.ptr
    %6 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<4 x i8>)>
    %7 = llvm.load %6 : !llvm.ptr -> i16
    llvm.switch %7 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%8: i32):  // pred: ^bb1
    %9 = llvm.call @"struct_construct<Tuple<u32>>"(%8) : (i32) -> !llvm.struct<packed (i32)>
    %10 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(%9) : (!llvm.struct<packed (i32)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %10 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb6(%11: i256):  // pred: ^bb2
    %12 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<packed (i32, i32, ptr)>
    %13 = llvm.call @"array_append<felt252>"(%12, %11) : (!llvm.struct<packed (i32, i32, ptr)>, i256) -> !llvm.struct<packed (i32, i32, ptr)>
    %14 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(%13) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %14 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: !llvm.struct<packed (i16, array<4 x i8>)>, %arg2: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(%arg1, %arg2) : (!llvm.struct<packed (i16, array<4 x i8>)>, i256) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(%arg0: !llvm.struct<packed (i16, array<8 x i8>)>, %arg1: i256) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<packed (i16, array<8 x i8>)>, i256)
  ^bb1:  // pred: ^bb4
    %0 = llvm.getelementptr inbounds %5[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i64)>
    %1 = llvm.load %0 : !llvm.ptr -> i64
    llvm.br ^bb5(%1 : i64)
  ^bb2:  // pred: ^bb4
    llvm.br ^bb6(%3 : i256)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%2: !llvm.struct<packed (i16, array<8 x i8>)>, %3: i256):  // pred: ^bb0
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %2, %5 : !llvm.struct<packed (i16, array<8 x i8>)>, !llvm.ptr
    %6 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    %7 = llvm.load %6 : !llvm.ptr -> i16
    llvm.switch %7 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%8: i64):  // pred: ^bb1
    %9 = llvm.call @"struct_construct<Tuple<u64>>"(%8) : (i64) -> !llvm.struct<packed (i64)>
    %10 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(%9) : (!llvm.struct<packed (i64)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %10 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb6(%11: i256):  // pred: ^bb2
    %12 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<packed (i32, i32, ptr)>
    %13 = llvm.call @"array_append<felt252>"(%12, %11) : (!llvm.struct<packed (i32, i32, ptr)>, i256) -> !llvm.struct<packed (i32, i32, ptr)>
    %14 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(%13) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %14 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: !llvm.struct<packed (i16, array<8 x i8>)>, %arg2: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(%arg1, %arg2) : (!llvm.struct<packed (i16, array<8 x i8>)>, i256) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(%arg0: !llvm.struct<packed (i16, array<16 x i8>)>, %arg1: i256) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<packed (i16, array<16 x i8>)>, i256)
  ^bb1:  // pred: ^bb4
    %0 = llvm.getelementptr inbounds %5[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i128)>
    %1 = llvm.load %0 : !llvm.ptr -> i128
    llvm.br ^bb5(%1 : i128)
  ^bb2:  // pred: ^bb4
    llvm.br ^bb6(%3 : i256)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%2: !llvm.struct<packed (i16, array<16 x i8>)>, %3: i256):  // pred: ^bb0
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %2, %5 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %6 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %7 = llvm.load %6 : !llvm.ptr -> i16
    llvm.switch %7 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%8: i128):  // pred: ^bb1
    %9 = llvm.call @"struct_construct<Tuple<u128>>"(%8) : (i128) -> !llvm.struct<packed (i128)>
    %10 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(%9) : (!llvm.struct<packed (i128)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %10 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb6(%11: i256):  // pred: ^bb2
    %12 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<packed (i32, i32, ptr)>
    %13 = llvm.call @"array_append<felt252>"(%12, %11) : (!llvm.struct<packed (i32, i32, ptr)>, i256) -> !llvm.struct<packed (i32, i32, ptr)>
    %14 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(%13) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %14 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: !llvm.struct<packed (i16, array<16 x i8>)>, %arg2: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(%arg1, %arg2) : (!llvm.struct<packed (i16, array<16 x i8>)>, i256) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
}
