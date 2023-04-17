module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"struct_deconstruct<Tuple<u16>>"(%arg0: !llvm.struct<(i16)>) -> i16 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16)> 
    llvm.return %0 : i16
  }
  llvm.func internal @"struct_deconstruct<Tuple<u32>>"(%arg0: !llvm.struct<(i32)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_deconstruct<Tuple<u64>>"(%arg0: !llvm.struct<(i64)>) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i64)> 
    llvm.return %0 : i64
  }
  llvm.func internal @"struct_deconstruct<Tuple<u128>>"(%arg0: !llvm.struct<(i128)>) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i128)> 
    llvm.return %0 : i128
  }
  llvm.func internal @"struct_construct<Tuple<u16, u16, u16>>"(%arg0: i16, %arg1: i16, %arg2: i16) -> !llvm.struct<(i16, i16, i16)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i16, i16, i16)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i16, i16, i16)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i16, i16, i16)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(i16, i16, i16)> 
    llvm.return %3 : !llvm.struct<(i16, i16, i16)>
  }
  llvm.func internal @"struct_construct<Tuple<u32, u32, u32>>"(%arg0: i32, %arg1: i32, %arg2: i32) -> !llvm.struct<(i32, i32, i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, i32, i32)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i32, i32, i32)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(i32, i32, i32)> 
    llvm.return %3 : !llvm.struct<(i32, i32, i32)>
  }
  llvm.func internal @"struct_construct<Tuple<u64, u64, u64>>"(%arg0: i64, %arg1: i64, %arg2: i64) -> !llvm.struct<(i64, i64, i64)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, i64, i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, i64, i64)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, i64, i64)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(i64, i64, i64)> 
    llvm.return %3 : !llvm.struct<(i64, i64, i64)>
  }
  llvm.func internal @"struct_construct<Tuple<u128, u128, u128>>"(%arg0: i128, %arg1: i128, %arg2: i128) -> !llvm.struct<(i128, i128, i128)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i128, i128, i128)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i128, i128, i128)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i128, i128, i128)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(i128, i128, i128)> 
    llvm.return %3 : !llvm.struct<(i128, i128, i128)>
  }
  llvm.func internal @"struct_construct<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>"(%arg0: !llvm.struct<(i16, i16, i16)>, %arg1: !llvm.struct<(i32, i32, i32)>, %arg2: !llvm.struct<(i64, i64, i64)>, %arg3: !llvm.struct<(i128, i128, i128)>) -> !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)> 
    %4 = llvm.insertvalue %arg3, %3[3] : !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)> 
    llvm.return %4 : !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>
  }
  llvm.func internal @"struct_construct<Tuple<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>>"(%arg0: !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>) -> !llvm.struct<(struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>)> 
    llvm.return %1 : !llvm.struct<(struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 0>"(%arg0: !llvm.struct<(struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>)>) -> !llvm.struct<(i16, array<90 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<90 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<90 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<90 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<90 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(13 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[116, 114, 97, 112, 32, 114, 101, 97, 99, 104, 101, 100, 0]> : tensor<13xi8>) : !llvm.array<13 x i8>
    llvm.store %2, %1 : !llvm.array<13 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1) : (i32, !llvm.ptr) -> i32
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 0>"(%arg0: i16) -> !llvm.struct<(i16, array<2 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<2 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<2 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<2 x i8>)>
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<2 x i8>)>
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 1>"(%arg0: i16) -> !llvm.struct<(i16, array<2 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<2 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<2 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<2 x i8>)>
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<2 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<u16>>"(%arg0: i16) -> !llvm.struct<(i16)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i16)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i16)> 
    llvm.return %1 : !llvm.struct<(i16)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(%arg0: !llvm.struct<(i16)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i16)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(13 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[116, 114, 97, 112, 32, 114, 101, 97, 99, 104, 101, 100, 0]> : tensor<13xi8>) : !llvm.array<13 x i8>
    llvm.store %2, %1 : !llvm.array<13 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1) : (i32, !llvm.ptr) -> i32
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 0>"(%arg0: i32) -> !llvm.struct<(i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<4 x i8>)>
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 1>"(%arg0: i32) -> !llvm.struct<(i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<4 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<u32>>"(%arg0: i32) -> !llvm.struct<(i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32)> 
    llvm.return %1 : !llvm.struct<(i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(%arg0: !llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(13 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[116, 114, 97, 112, 32, 114, 101, 97, 99, 104, 101, 100, 0]> : tensor<13xi8>) : !llvm.array<13 x i8>
    llvm.store %2, %1 : !llvm.array<13 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1) : (i32, !llvm.ptr) -> i32
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 0>"(%arg0: i64) -> !llvm.struct<(i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %arg0, %4 : i64, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 1>"(%arg0: i64) -> !llvm.struct<(i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %arg0, %4 : i64, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<u64>>"(%arg0: i64) -> !llvm.struct<(i64)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64)> 
    llvm.return %1 : !llvm.struct<(i64)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(%arg0: !llvm.struct<(i64)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i64)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(13 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[116, 114, 97, 112, 32, 114, 101, 97, 99, 104, 101, 100, 0]> : tensor<13xi8>) : !llvm.array<13 x i8>
    llvm.store %2, %1 : !llvm.array<13 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1) : (i32, !llvm.ptr) -> i32
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 0>"(%arg0: i128) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : i128, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 1>"(%arg0: i128) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : i128, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<u128>>"(%arg0: i128) -> !llvm.struct<(i128)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i128)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i128)> 
    llvm.return %1 : !llvm.struct<(i128)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(%arg0: !llvm.struct<(i128)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i128)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(13 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[116, 114, 97, 112, 32, 114, 101, 97, 99, 104, 101, 100, 0]> : tensor<13xi8>) : !llvm.array<13 x i8>
    llvm.store %2, %1 : !llvm.array<13 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1) : (i32, !llvm.ptr) -> i32
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"array_new<felt252>"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(i32, i32, ptr)> 
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"array_append<felt252>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i256) -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<(i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<(i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<(i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<(i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<(i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    llvm.store %arg1, %12 : i256, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func @"uint_addition::uint_addition::main"() -> !llvm.struct<(i16, array<90 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb37
  ^bb1:  // pred: ^bb37
    %0 = llvm.load %30 : !llvm.ptr -> !llvm.struct<(i16)>
    llvm.br ^bb38(%0 : !llvm.struct<(i16)>)
  ^bb2:  // pred: ^bb37
    %1 = llvm.load %30 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb61(%1 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb37
    llvm.unreachable
  ^bb4:  // pred: ^bb38
    %2 = llvm.load %39 : !llvm.ptr -> !llvm.struct<(i16)>
    llvm.br ^bb39(%32, %2 : i16, !llvm.struct<(i16)>)
  ^bb5:  // pred: ^bb38
    %3 = llvm.load %39 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb60(%3 : !llvm.struct<(i32, i32, ptr)>)
  ^bb6:  // pred: ^bb38
    llvm.unreachable
  ^bb7:  // pred: ^bb39
    %4 = llvm.load %49 : !llvm.ptr -> !llvm.struct<(i16)>
    llvm.br ^bb40(%40, %42, %4 : i16, i16, !llvm.struct<(i16)>)
  ^bb8:  // pred: ^bb39
    %5 = llvm.load %49 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb59(%5 : !llvm.struct<(i32, i32, ptr)>)
  ^bb9:  // pred: ^bb39
    llvm.unreachable
  ^bb10:  // pred: ^bb40
    %6 = llvm.load %60 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb41(%50, %51, %53, %6 : i16, i16, i16, !llvm.struct<(i32)>)
  ^bb11:  // pred: ^bb40
    %7 = llvm.load %60 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb58(%7 : !llvm.struct<(i32, i32, ptr)>)
  ^bb12:  // pred: ^bb40
    llvm.unreachable
  ^bb13:  // pred: ^bb41
    %8 = llvm.load %72 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb42(%61, %62, %63, %65, %8 : i16, i16, i16, i32, !llvm.struct<(i32)>)
  ^bb14:  // pred: ^bb41
    %9 = llvm.load %72 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb57(%9 : !llvm.struct<(i32, i32, ptr)>)
  ^bb15:  // pred: ^bb41
    llvm.unreachable
  ^bb16:  // pred: ^bb42
    %10 = llvm.load %85 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb43(%73, %74, %75, %76, %78, %10 : i16, i16, i16, i32, i32, !llvm.struct<(i32)>)
  ^bb17:  // pred: ^bb42
    %11 = llvm.load %85 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb56(%11 : !llvm.struct<(i32, i32, ptr)>)
  ^bb18:  // pred: ^bb42
    llvm.unreachable
  ^bb19:  // pred: ^bb43
    %12 = llvm.load %99 : !llvm.ptr -> !llvm.struct<(i64)>
    llvm.br ^bb44(%86, %87, %88, %89, %90, %92, %12 : i16, i16, i16, i32, i32, i32, !llvm.struct<(i64)>)
  ^bb20:  // pred: ^bb43
    %13 = llvm.load %99 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb55(%13 : !llvm.struct<(i32, i32, ptr)>)
  ^bb21:  // pred: ^bb43
    llvm.unreachable
  ^bb22:  // pred: ^bb44
    %14 = llvm.load %114 : !llvm.ptr -> !llvm.struct<(i64)>
    llvm.br ^bb45(%100, %101, %102, %103, %104, %105, %107, %14 : i16, i16, i16, i32, i32, i32, i64, !llvm.struct<(i64)>)
  ^bb23:  // pred: ^bb44
    %15 = llvm.load %114 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb54(%15 : !llvm.struct<(i32, i32, ptr)>)
  ^bb24:  // pred: ^bb44
    llvm.unreachable
  ^bb25:  // pred: ^bb45
    %16 = llvm.load %130 : !llvm.ptr -> !llvm.struct<(i64)>
    llvm.br ^bb46(%115, %116, %117, %118, %119, %120, %121, %123, %16 : i16, i16, i16, i32, i32, i32, i64, i64, !llvm.struct<(i64)>)
  ^bb26:  // pred: ^bb45
    %17 = llvm.load %130 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb53(%17 : !llvm.struct<(i32, i32, ptr)>)
  ^bb27:  // pred: ^bb45
    llvm.unreachable
  ^bb28:  // pred: ^bb46
    %18 = llvm.load %147 : !llvm.ptr -> !llvm.struct<(i128)>
    llvm.br ^bb47(%131, %132, %133, %134, %135, %136, %137, %138, %140, %18 : i16, i16, i16, i32, i32, i32, i64, i64, i64, !llvm.struct<(i128)>)
  ^bb29:  // pred: ^bb46
    %19 = llvm.load %147 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb52(%19 : !llvm.struct<(i32, i32, ptr)>)
  ^bb30:  // pred: ^bb46
    llvm.unreachable
  ^bb31:  // pred: ^bb47
    %20 = llvm.load %165 : !llvm.ptr -> !llvm.struct<(i128)>
    llvm.br ^bb48(%148, %149, %150, %151, %152, %153, %154, %155, %156, %158, %20 : i16, i16, i16, i32, i32, i32, i64, i64, i64, i128, !llvm.struct<(i128)>)
  ^bb32:  // pred: ^bb47
    %21 = llvm.load %165 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb51(%21 : !llvm.struct<(i32, i32, ptr)>)
  ^bb33:  // pred: ^bb47
    llvm.unreachable
  ^bb34:  // pred: ^bb48
    %22 = llvm.load %184 : !llvm.ptr -> !llvm.struct<(i128)>
    llvm.br ^bb49(%166, %167, %168, %169, %170, %171, %172, %173, %174, %175, %177, %22 : i16, i16, i16, i32, i32, i32, i64, i64, i64, i128, i128, !llvm.struct<(i128)>)
  ^bb35:  // pred: ^bb48
    %23 = llvm.load %184 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb50(%23 : !llvm.struct<(i32, i32, ptr)>)
  ^bb36:  // pred: ^bb48
    llvm.unreachable
  ^bb37:  // pred: ^bb0
    %24 = llvm.mlir.constant(4 : i16) : i16
    %25 = llvm.mlir.constant(6 : i16) : i16
    %26 = llvm.call @"core::integer::U16Add::add"(%24, %25) : (i16, i16) -> !llvm.struct<(i16, array<16 x i8>)>
    %27 = llvm.extractvalue %26[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %28 = llvm.extractvalue %26[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %29 = llvm.mlir.constant(1 : i64) : i64
    %30 = llvm.alloca %29 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %28, %30 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %27 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb38(%31: !llvm.struct<(i16)>):  // pred: ^bb1
    %32 = llvm.call @"struct_deconstruct<Tuple<u16>>"(%31) : (!llvm.struct<(i16)>) -> i16
    %33 = llvm.mlir.constant(2 : i16) : i16
    %34 = llvm.mlir.constant(10 : i16) : i16
    %35 = llvm.call @"core::integer::U16Add::add"(%33, %34) : (i16, i16) -> !llvm.struct<(i16, array<16 x i8>)>
    %36 = llvm.extractvalue %35[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %37 = llvm.extractvalue %35[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %38 = llvm.mlir.constant(1 : i64) : i64
    %39 = llvm.alloca %38 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %37, %39 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %36 : i16, ^bb6 [
      0: ^bb4,
      1: ^bb5
    ]
  ^bb39(%40: i16, %41: !llvm.struct<(i16)>):  // pred: ^bb4
    %42 = llvm.call @"struct_deconstruct<Tuple<u16>>"(%41) : (!llvm.struct<(i16)>) -> i16
    %43 = llvm.mlir.constant(50 : i16) : i16
    %44 = llvm.mlir.constant(2 : i16) : i16
    %45 = llvm.call @"core::integer::U16Add::add"(%43, %44) : (i16, i16) -> !llvm.struct<(i16, array<16 x i8>)>
    %46 = llvm.extractvalue %45[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %47 = llvm.extractvalue %45[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %48 = llvm.mlir.constant(1 : i64) : i64
    %49 = llvm.alloca %48 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %47, %49 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %46 : i16, ^bb9 [
      0: ^bb7,
      1: ^bb8
    ]
  ^bb40(%50: i16, %51: i16, %52: !llvm.struct<(i16)>):  // pred: ^bb7
    %53 = llvm.call @"struct_deconstruct<Tuple<u16>>"(%52) : (!llvm.struct<(i16)>) -> i16
    %54 = llvm.mlir.constant(4 : i32) : i32
    %55 = llvm.mlir.constant(6 : i32) : i32
    %56 = llvm.call @"core::integer::U32Add::add"(%54, %55) : (i32, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %57 = llvm.extractvalue %56[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %58 = llvm.extractvalue %56[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %59 = llvm.mlir.constant(1 : i64) : i64
    %60 = llvm.alloca %59 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %58, %60 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %57 : i16, ^bb12 [
      0: ^bb10,
      1: ^bb11
    ]
  ^bb41(%61: i16, %62: i16, %63: i16, %64: !llvm.struct<(i32)>):  // pred: ^bb10
    %65 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%64) : (!llvm.struct<(i32)>) -> i32
    %66 = llvm.mlir.constant(2 : i32) : i32
    %67 = llvm.mlir.constant(10 : i32) : i32
    %68 = llvm.call @"core::integer::U32Add::add"(%66, %67) : (i32, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %69 = llvm.extractvalue %68[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %70 = llvm.extractvalue %68[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %71 = llvm.mlir.constant(1 : i64) : i64
    %72 = llvm.alloca %71 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %70, %72 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %69 : i16, ^bb15 [
      0: ^bb13,
      1: ^bb14
    ]
  ^bb42(%73: i16, %74: i16, %75: i16, %76: i32, %77: !llvm.struct<(i32)>):  // pred: ^bb13
    %78 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%77) : (!llvm.struct<(i32)>) -> i32
    %79 = llvm.mlir.constant(50 : i32) : i32
    %80 = llvm.mlir.constant(2 : i32) : i32
    %81 = llvm.call @"core::integer::U32Add::add"(%79, %80) : (i32, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %82 = llvm.extractvalue %81[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %83 = llvm.extractvalue %81[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %84 = llvm.mlir.constant(1 : i64) : i64
    %85 = llvm.alloca %84 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %83, %85 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %82 : i16, ^bb18 [
      0: ^bb16,
      1: ^bb17
    ]
  ^bb43(%86: i16, %87: i16, %88: i16, %89: i32, %90: i32, %91: !llvm.struct<(i32)>):  // pred: ^bb16
    %92 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%91) : (!llvm.struct<(i32)>) -> i32
    %93 = llvm.mlir.constant(4 : i64) : i64
    %94 = llvm.mlir.constant(6 : i64) : i64
    %95 = llvm.call @"core::integer::U64Add::add"(%93, %94) : (i64, i64) -> !llvm.struct<(i16, array<16 x i8>)>
    %96 = llvm.extractvalue %95[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %97 = llvm.extractvalue %95[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %98 = llvm.mlir.constant(1 : i64) : i64
    %99 = llvm.alloca %98 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %97, %99 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %96 : i16, ^bb21 [
      0: ^bb19,
      1: ^bb20
    ]
  ^bb44(%100: i16, %101: i16, %102: i16, %103: i32, %104: i32, %105: i32, %106: !llvm.struct<(i64)>):  // pred: ^bb19
    %107 = llvm.call @"struct_deconstruct<Tuple<u64>>"(%106) : (!llvm.struct<(i64)>) -> i64
    %108 = llvm.mlir.constant(2 : i64) : i64
    %109 = llvm.mlir.constant(10 : i64) : i64
    %110 = llvm.call @"core::integer::U64Add::add"(%108, %109) : (i64, i64) -> !llvm.struct<(i16, array<16 x i8>)>
    %111 = llvm.extractvalue %110[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %112 = llvm.extractvalue %110[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %113 = llvm.mlir.constant(1 : i64) : i64
    %114 = llvm.alloca %113 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %112, %114 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %111 : i16, ^bb24 [
      0: ^bb22,
      1: ^bb23
    ]
  ^bb45(%115: i16, %116: i16, %117: i16, %118: i32, %119: i32, %120: i32, %121: i64, %122: !llvm.struct<(i64)>):  // pred: ^bb22
    %123 = llvm.call @"struct_deconstruct<Tuple<u64>>"(%122) : (!llvm.struct<(i64)>) -> i64
    %124 = llvm.mlir.constant(50 : i64) : i64
    %125 = llvm.mlir.constant(2 : i64) : i64
    %126 = llvm.call @"core::integer::U64Add::add"(%124, %125) : (i64, i64) -> !llvm.struct<(i16, array<16 x i8>)>
    %127 = llvm.extractvalue %126[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %128 = llvm.extractvalue %126[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %129 = llvm.mlir.constant(1 : i64) : i64
    %130 = llvm.alloca %129 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %128, %130 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %127 : i16, ^bb27 [
      0: ^bb25,
      1: ^bb26
    ]
  ^bb46(%131: i16, %132: i16, %133: i16, %134: i32, %135: i32, %136: i32, %137: i64, %138: i64, %139: !llvm.struct<(i64)>):  // pred: ^bb25
    %140 = llvm.call @"struct_deconstruct<Tuple<u64>>"(%139) : (!llvm.struct<(i64)>) -> i64
    %141 = llvm.mlir.constant(4 : i128) : i128
    %142 = llvm.mlir.constant(6 : i128) : i128
    %143 = llvm.call @"core::integer::U128Add::add"(%141, %142) : (i128, i128) -> !llvm.struct<(i16, array<16 x i8>)>
    %144 = llvm.extractvalue %143[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %145 = llvm.extractvalue %143[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %146 = llvm.mlir.constant(1 : i64) : i64
    %147 = llvm.alloca %146 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %145, %147 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %144 : i16, ^bb30 [
      0: ^bb28,
      1: ^bb29
    ]
  ^bb47(%148: i16, %149: i16, %150: i16, %151: i32, %152: i32, %153: i32, %154: i64, %155: i64, %156: i64, %157: !llvm.struct<(i128)>):  // pred: ^bb28
    %158 = llvm.call @"struct_deconstruct<Tuple<u128>>"(%157) : (!llvm.struct<(i128)>) -> i128
    %159 = llvm.mlir.constant(2 : i128) : i128
    %160 = llvm.mlir.constant(10 : i128) : i128
    %161 = llvm.call @"core::integer::U128Add::add"(%159, %160) : (i128, i128) -> !llvm.struct<(i16, array<16 x i8>)>
    %162 = llvm.extractvalue %161[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %163 = llvm.extractvalue %161[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %164 = llvm.mlir.constant(1 : i64) : i64
    %165 = llvm.alloca %164 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %163, %165 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %162 : i16, ^bb33 [
      0: ^bb31,
      1: ^bb32
    ]
  ^bb48(%166: i16, %167: i16, %168: i16, %169: i32, %170: i32, %171: i32, %172: i64, %173: i64, %174: i64, %175: i128, %176: !llvm.struct<(i128)>):  // pred: ^bb31
    %177 = llvm.call @"struct_deconstruct<Tuple<u128>>"(%176) : (!llvm.struct<(i128)>) -> i128
    %178 = llvm.mlir.constant(50 : i128) : i128
    %179 = llvm.mlir.constant(2 : i128) : i128
    %180 = llvm.call @"core::integer::U128Add::add"(%178, %179) : (i128, i128) -> !llvm.struct<(i16, array<16 x i8>)>
    %181 = llvm.extractvalue %180[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %182 = llvm.extractvalue %180[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %183 = llvm.mlir.constant(1 : i64) : i64
    %184 = llvm.alloca %183 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %182, %184 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %181 : i16, ^bb36 [
      0: ^bb34,
      1: ^bb35
    ]
  ^bb49(%185: i16, %186: i16, %187: i16, %188: i32, %189: i32, %190: i32, %191: i64, %192: i64, %193: i64, %194: i128, %195: i128, %196: !llvm.struct<(i128)>):  // pred: ^bb34
    %197 = llvm.call @"struct_deconstruct<Tuple<u128>>"(%196) : (!llvm.struct<(i128)>) -> i128
    %198 = llvm.call @"struct_construct<Tuple<u16, u16, u16>>"(%185, %186, %187) : (i16, i16, i16) -> !llvm.struct<(i16, i16, i16)>
    %199 = llvm.call @"struct_construct<Tuple<u32, u32, u32>>"(%188, %189, %190) : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32)>
    %200 = llvm.call @"struct_construct<Tuple<u64, u64, u64>>"(%191, %192, %193) : (i64, i64, i64) -> !llvm.struct<(i64, i64, i64)>
    %201 = llvm.call @"struct_construct<Tuple<u128, u128, u128>>"(%194, %195, %197) : (i128, i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %202 = llvm.call @"struct_construct<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>"(%198, %199, %200, %201) : (!llvm.struct<(i16, i16, i16)>, !llvm.struct<(i32, i32, i32)>, !llvm.struct<(i64, i64, i64)>, !llvm.struct<(i128, i128, i128)>) -> !llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>
    %203 = llvm.call @"struct_construct<Tuple<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>>"(%202) : (!llvm.struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>) -> !llvm.struct<(struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>)>
    %204 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 0>"(%203) : (!llvm.struct<(struct<(struct<(i16, i16, i16)>, struct<(i32, i32, i32)>, struct<(i64, i64, i64)>, struct<(i128, i128, i128)>)>)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %204 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb50(%205: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb35
    %206 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%205) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %206 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb51(%207: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb32
    %208 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%207) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %208 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb52(%209: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb29
    %210 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%209) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %210 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb53(%211: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb26
    %212 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%211) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %212 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb54(%213: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb23
    %214 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%213) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %214 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb55(%215: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb20
    %216 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%215) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %216 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb56(%217: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb17
    %218 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%217) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %218 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb57(%219: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb14
    %220 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%219) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %220 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb58(%221: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb11
    %222 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%221) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %222 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb59(%223: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb8
    %224 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%223) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %224 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb60(%225: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb5
    %226 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%225) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %226 : !llvm.struct<(i16, array<90 x i8>)>
  ^bb61(%227: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %228 = llvm.call @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(%227) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.return %228 : !llvm.struct<(i16, array<90 x i8>)>
  }
  llvm.func @"_mlir_ciface_uint_addition::uint_addition::main"(%arg0: !llvm.ptr<struct<(i16, array<90 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"uint_addition::uint_addition::main"() : () -> !llvm.struct<(i16, array<90 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<90 x i8>)>>
    llvm.return
  }
  llvm.func @"core::integer::U16Add::add"(%arg0: i16, %arg1: i16) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : i16, i16)
  ^bb1:  // pred: ^bb7
    %0 = llvm.load %17 : !llvm.ptr -> !llvm.struct<(i16)>
    llvm.br ^bb8(%0 : !llvm.struct<(i16)>)
  ^bb2:  // pred: ^bb7
    %1 = llvm.load %17 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb9(%1 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb7
    llvm.unreachable
  ^bb4(%2: i16, %3: i16):  // pred: ^bb0
    %4 = "llvm.intr.uadd.with.overflow"(%2, %3) : (i16, i16) -> !llvm.struct<(i16, i1)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i16, i1)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(i16, i1)> 
    llvm.cond_br %6, ^bb6(%5 : i16), ^bb5(%5 : i16)
  ^bb5(%7: i16):  // pred: ^bb4
    %8 = llvm.call @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 0>"(%7) : (i16) -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.br ^bb7(%8 : !llvm.struct<(i16, array<2 x i8>)>)
  ^bb6(%9: i16):  // pred: ^bb4
    %10 = llvm.call @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 1>"(%9) : (i16) -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.br ^bb7(%10 : !llvm.struct<(i16, array<2 x i8>)>)
  ^bb7(%11: !llvm.struct<(i16, array<2 x i8>)>):  // 2 preds: ^bb5, ^bb6
    %12 = llvm.mlir.constant(155775200859838811096160292336445452151 : i256) : i256
    %13 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(%11, %12) : (!llvm.struct<(i16, array<2 x i8>)>, i256) -> !llvm.struct<(i16, array<16 x i8>)>
    %14 = llvm.extractvalue %13[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %14 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb8(%18: !llvm.struct<(i16)>):  // pred: ^bb1
    %19 = llvm.call @"struct_deconstruct<Tuple<u16>>"(%18) : (!llvm.struct<(i16)>) -> i16
    %20 = llvm.call @"struct_construct<Tuple<u16>>"(%19) : (i16) -> !llvm.struct<(i16)>
    %21 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(%20) : (!llvm.struct<(i16)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %21 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb9(%22: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %23 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(%22) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %23 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U16Add::add"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: i16, %arg2: i16) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::integer::U16Add::add"(%arg1, %arg2) : (i16, i16) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::integer::U32Add::add"(%arg0: i32, %arg1: i32) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : i32, i32)
  ^bb1:  // pred: ^bb7
    %0 = llvm.load %17 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb8(%0 : !llvm.struct<(i32)>)
  ^bb2:  // pred: ^bb7
    %1 = llvm.load %17 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb9(%1 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb7
    llvm.unreachable
  ^bb4(%2: i32, %3: i32):  // pred: ^bb0
    %4 = "llvm.intr.uadd.with.overflow"(%2, %3) : (i32, i32) -> !llvm.struct<(i32, i1)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i32, i1)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(i32, i1)> 
    llvm.cond_br %6, ^bb6(%5 : i32), ^bb5(%5 : i32)
  ^bb5(%7: i32):  // pred: ^bb4
    %8 = llvm.call @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 0>"(%7) : (i32) -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.br ^bb7(%8 : !llvm.struct<(i16, array<4 x i8>)>)
  ^bb6(%9: i32):  // pred: ^bb4
    %10 = llvm.call @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 1>"(%9) : (i32) -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.br ^bb7(%10 : !llvm.struct<(i16, array<4 x i8>)>)
  ^bb7(%11: !llvm.struct<(i16, array<4 x i8>)>):  // 2 preds: ^bb5, ^bb6
    %12 = llvm.mlir.constant(155785504323917466144735657540098748279 : i256) : i256
    %13 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(%11, %12) : (!llvm.struct<(i16, array<4 x i8>)>, i256) -> !llvm.struct<(i16, array<16 x i8>)>
    %14 = llvm.extractvalue %13[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %14 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb8(%18: !llvm.struct<(i32)>):  // pred: ^bb1
    %19 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%18) : (!llvm.struct<(i32)>) -> i32
    %20 = llvm.call @"struct_construct<Tuple<u32>>"(%19) : (i32) -> !llvm.struct<(i32)>
    %21 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(%20) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %21 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb9(%22: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %23 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(%22) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %23 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U32Add::add"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: i32, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::integer::U32Add::add"(%arg1, %arg2) : (i32, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::integer::U64Add::add"(%arg0: i64, %arg1: i64) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : i64, i64)
  ^bb1:  // pred: ^bb7
    %0 = llvm.load %17 : !llvm.ptr -> !llvm.struct<(i64)>
    llvm.br ^bb8(%0 : !llvm.struct<(i64)>)
  ^bb2:  // pred: ^bb7
    %1 = llvm.load %17 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb9(%1 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb7
    llvm.unreachable
  ^bb4(%2: i64, %3: i64):  // pred: ^bb0
    %4 = "llvm.intr.uadd.with.overflow"(%2, %3) : (i64, i64) -> !llvm.struct<(i64, i1)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i64, i1)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(i64, i1)> 
    llvm.cond_br %6, ^bb6(%5 : i64), ^bb5(%5 : i64)
  ^bb5(%7: i64):  // pred: ^bb4
    %8 = llvm.call @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 0>"(%7) : (i64) -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.br ^bb7(%8 : !llvm.struct<(i16, array<8 x i8>)>)
  ^bb6(%9: i64):  // pred: ^bb4
    %10 = llvm.call @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 1>"(%9) : (i64) -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.br ^bb7(%10 : !llvm.struct<(i16, array<8 x i8>)>)
  ^bb7(%11: !llvm.struct<(i16, array<8 x i8>)>):  // 2 preds: ^bb5, ^bb6
    %12 = llvm.mlir.constant(155801121779312277930962096923588980599 : i256) : i256
    %13 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(%11, %12) : (!llvm.struct<(i16, array<8 x i8>)>, i256) -> !llvm.struct<(i16, array<16 x i8>)>
    %14 = llvm.extractvalue %13[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %14 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb8(%18: !llvm.struct<(i64)>):  // pred: ^bb1
    %19 = llvm.call @"struct_deconstruct<Tuple<u64>>"(%18) : (!llvm.struct<(i64)>) -> i64
    %20 = llvm.call @"struct_construct<Tuple<u64>>"(%19) : (i64) -> !llvm.struct<(i64)>
    %21 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(%20) : (!llvm.struct<(i64)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %21 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb9(%22: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %23 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(%22) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %23 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U64Add::add"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: i64, %arg2: i64) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::integer::U64Add::add"(%arg1, %arg2) : (i64, i64) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::integer::U128Add::add"(%arg0: i128, %arg1: i128) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : i128, i128)
  ^bb1:  // pred: ^bb7
    %0 = llvm.load %17 : !llvm.ptr -> !llvm.struct<(i128)>
    llvm.br ^bb8(%0 : !llvm.struct<(i128)>)
  ^bb2:  // pred: ^bb7
    %1 = llvm.load %17 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb9(%1 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb7
    llvm.unreachable
  ^bb4(%2: i128, %3: i128):  // pred: ^bb0
    %4 = "llvm.intr.uadd.with.overflow"(%2, %3) : (i128, i128) -> !llvm.struct<(i128, i1)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i128, i1)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(i128, i1)> 
    llvm.cond_br %6, ^bb6(%5 : i128), ^bb5(%5 : i128)
  ^bb5(%7: i128):  // pred: ^bb4
    %8 = llvm.call @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 0>"(%7) : (i128) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.br ^bb7(%8 : !llvm.struct<(i16, array<16 x i8>)>)
  ^bb6(%9: i128):  // pred: ^bb4
    %10 = llvm.call @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 1>"(%9) : (i128) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.br ^bb7(%10 : !llvm.struct<(i16, array<16 x i8>)>)
  ^bb7(%11: !llvm.struct<(i16, array<16 x i8>)>):  // 2 preds: ^bb5, ^bb6
    %12 = llvm.mlir.constant(39878429859757942499084499860145094553463 : i256) : i256
    %13 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(%11, %12) : (!llvm.struct<(i16, array<16 x i8>)>, i256) -> !llvm.struct<(i16, array<16 x i8>)>
    %14 = llvm.extractvalue %13[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.alloca %16 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %14 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb8(%18: !llvm.struct<(i128)>):  // pred: ^bb1
    %19 = llvm.call @"struct_deconstruct<Tuple<u128>>"(%18) : (!llvm.struct<(i128)>) -> i128
    %20 = llvm.call @"struct_construct<Tuple<u128>>"(%19) : (i128) -> !llvm.struct<(i128)>
    %21 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(%20) : (!llvm.struct<(i128)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %21 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb9(%22: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %23 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(%22) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %23 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::integer::U128Add::add"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: i128, %arg2: i128) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::integer::U128Add::add"(%arg1, %arg2) : (i128, i128) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(%arg0: !llvm.struct<(i16, array<2 x i8>)>, %arg1: i256) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<(i16, array<2 x i8>)>, i256)
  ^bb1:  // pred: ^bb4
    %0 = llvm.load %6 : !llvm.ptr -> i16
    llvm.br ^bb5(%0 : i16)
  ^bb2:  // pred: ^bb4
    llvm.br ^bb6(%2 : i256)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%1: !llvm.struct<(i16, array<2 x i8>)>, %2: i256):  // pred: ^bb0
    %3 = llvm.extractvalue %1[0] : !llvm.struct<(i16, array<2 x i8>)> 
    %4 = llvm.extractvalue %1[1] : !llvm.struct<(i16, array<2 x i8>)> 
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.array<2 x i8> : (i64) -> !llvm.ptr
    llvm.store %4, %6 : !llvm.array<2 x i8>, !llvm.ptr
    llvm.switch %3 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%7: i16):  // pred: ^bb1
    %8 = llvm.call @"struct_construct<Tuple<u16>>"(%7) : (i16) -> !llvm.struct<(i16)>
    %9 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(%8) : (!llvm.struct<(i16)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %9 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb6(%10: i256):  // pred: ^bb2
    %11 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %12 = llvm.call @"array_append<felt252>"(%11, %10) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(%12) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %13 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: !llvm.struct<(i16, array<2 x i8>)>, %arg2: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(%arg1, %arg2) : (!llvm.struct<(i16, array<2 x i8>)>, i256) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(%arg0: !llvm.struct<(i16, array<4 x i8>)>, %arg1: i256) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<(i16, array<4 x i8>)>, i256)
  ^bb1:  // pred: ^bb4
    %0 = llvm.load %6 : !llvm.ptr -> i32
    llvm.br ^bb5(%0 : i32)
  ^bb2:  // pred: ^bb4
    llvm.br ^bb6(%2 : i256)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%1: !llvm.struct<(i16, array<4 x i8>)>, %2: i256):  // pred: ^bb0
    %3 = llvm.extractvalue %1[0] : !llvm.struct<(i16, array<4 x i8>)> 
    %4 = llvm.extractvalue %1[1] : !llvm.struct<(i16, array<4 x i8>)> 
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.array<4 x i8> : (i64) -> !llvm.ptr
    llvm.store %4, %6 : !llvm.array<4 x i8>, !llvm.ptr
    llvm.switch %3 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%7: i32):  // pred: ^bb1
    %8 = llvm.call @"struct_construct<Tuple<u32>>"(%7) : (i32) -> !llvm.struct<(i32)>
    %9 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(%8) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %9 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb6(%10: i256):  // pred: ^bb2
    %11 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %12 = llvm.call @"array_append<felt252>"(%11, %10) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(%12) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %13 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: !llvm.struct<(i16, array<4 x i8>)>, %arg2: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(%arg1, %arg2) : (!llvm.struct<(i16, array<4 x i8>)>, i256) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(%arg0: !llvm.struct<(i16, array<8 x i8>)>, %arg1: i256) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<(i16, array<8 x i8>)>, i256)
  ^bb1:  // pred: ^bb4
    %0 = llvm.load %6 : !llvm.ptr -> i64
    llvm.br ^bb5(%0 : i64)
  ^bb2:  // pred: ^bb4
    llvm.br ^bb6(%2 : i256)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%1: !llvm.struct<(i16, array<8 x i8>)>, %2: i256):  // pred: ^bb0
    %3 = llvm.extractvalue %1[0] : !llvm.struct<(i16, array<8 x i8>)> 
    %4 = llvm.extractvalue %1[1] : !llvm.struct<(i16, array<8 x i8>)> 
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.array<8 x i8> : (i64) -> !llvm.ptr
    llvm.store %4, %6 : !llvm.array<8 x i8>, !llvm.ptr
    llvm.switch %3 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%7: i64):  // pred: ^bb1
    %8 = llvm.call @"struct_construct<Tuple<u64>>"(%7) : (i64) -> !llvm.struct<(i64)>
    %9 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(%8) : (!llvm.struct<(i64)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %9 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb6(%10: i256):  // pred: ^bb2
    %11 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %12 = llvm.call @"array_append<felt252>"(%11, %10) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(%12) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %13 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: !llvm.struct<(i16, array<8 x i8>)>, %arg2: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(%arg1, %arg2) : (!llvm.struct<(i16, array<8 x i8>)>, i256) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(%arg0: !llvm.struct<(i16, array<16 x i8>)>, %arg1: i256) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<(i16, array<16 x i8>)>, i256)
  ^bb1:  // pred: ^bb4
    %0 = llvm.load %6 : !llvm.ptr -> i128
    llvm.br ^bb5(%0 : i128)
  ^bb2:  // pred: ^bb4
    llvm.br ^bb6(%2 : i256)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%1: !llvm.struct<(i16, array<16 x i8>)>, %2: i256):  // pred: ^bb0
    %3 = llvm.extractvalue %1[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %4 = llvm.extractvalue %1[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %4, %6 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %3 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%7: i128):  // pred: ^bb1
    %8 = llvm.call @"struct_construct<Tuple<u128>>"(%7) : (i128) -> !llvm.struct<(i128)>
    %9 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(%8) : (!llvm.struct<(i128)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %9 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb6(%10: i256):  // pred: ^bb2
    %11 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %12 = llvm.call @"array_append<felt252>"(%11, %10) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(%12) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %13 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: !llvm.struct<(i16, array<16 x i8>)>, %arg2: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(%arg1, %arg2) : (!llvm.struct<(i16, array<16 x i8>)>, i256) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
}
