main := fn(): int {
	a := 1;
	b := &a;
	modify(b);
	drop(a);
	stack_reclamation_edge_case := 0;
	return *b - 2;
}

modify := fn(a: ^int): void {
	*a = 2;
	return;
}

drop := fn(a: int): void {
	return;
}
