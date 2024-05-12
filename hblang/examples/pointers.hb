main := fn(): int {
	a := 1;
	b := &a;
	*b = 2;
	return a - 2;
}
