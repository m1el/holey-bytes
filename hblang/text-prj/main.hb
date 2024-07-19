foo := 0;
.{global, fib} := @use("pkg.hb")


main := fn(a: int): int {
	return fib(global)
}
