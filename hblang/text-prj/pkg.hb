global := 10

Structa := struct {
	foo: int,
	goo: int,
}

create_window := fn(): WindowID {
    return WindowID.(1, 2)
}

WindowID := struct {
    host_id: int,
    window_id: int,
}

fib := fn(n: int): int {
	return n + 1
}
