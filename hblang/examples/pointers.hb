main := fn(): int {
	a := 1;
	b := &a;
	modify(b);
	return a - 2;
}

modify := fn(a: *int): void {
	*a = 2;
	return;
}
