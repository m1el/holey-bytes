Vec := fn($Elem: type): type {
	return struct {
		data: ^Elem,
		len: uint,
		cap: uint,
	};
}

main := fn(): int {
	i := 69;
	vec := Vec(int).{
		data: &i,
		len: 1,
		cap: 1,
	};
	return *vec.data;
}
