Type := struct {
	brah: int,
	blah: int,
}

main := fn(): int {
	byte := @as(u8, 10);
	same_type_as_byte := @as(@TypeOf(byte), 30);
	wide_uint := @as(u32, 40);
	truncated_uint := @as(u8, @intcast(wide_uint));
	size_of_Type_in_bytes := @sizeof(Type);
	align_of_Type_in_bytes := @alignof(Type);
	hardcoded_pointer := @as(^u8, @bitcast(10));
	ecall_that_returns_int := @eca(int, 1, Type.(10, 20), 5, 6);
	return 0;
}
