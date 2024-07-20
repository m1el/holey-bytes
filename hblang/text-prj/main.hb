foo := 0;

.{global, fib, Structa, create_window, WindowID} := @use("pkg.hb")

main := fn(a: int): int {
	g := global

	win := create_window()

	return fib(g + Structa.(0, 0).foo)
}
