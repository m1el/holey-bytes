
Type := struct {
	brah: int,
	blah: int,
}

main := fn(): int {
	return @eca(int, 1, Type.(10, 20), @sizeof(Type), @alignof(Type), 5, 6);
}
