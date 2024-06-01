pkg := @use("pkg.hb");

main := fn(a: int): int {
	return pkg.fib(10);
}
