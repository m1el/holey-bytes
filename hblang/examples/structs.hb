Ty := struct {
	a: int,
	b: int,
}

main := fn(): int {
	inst := Ty.{ a: 1, b: 2 };
	return inst.a + inst.b;
}
