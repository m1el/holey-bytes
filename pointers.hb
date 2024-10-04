Point := struct {
	x: int,
	y: int,
}

Rect := struct {
	min: Point,
	max: Point,
}

main := fn(): void {
	rect := Rect.(.(0, 0), .(0, 0))
	// eliminates initila 0
	rect.min.x = 1
	// here as well
	rect.min.y = 2
	// eliminates previous 2 lines, intermidiate stack slot is created, and stores are
	// delegated to the rect
	rect.min = .(3, 4)

	// encompasses the previous two loads
	ptr := &rect.min
	// pointer escapes to a function -> rect.min now has unknown values
	clobber(ptr)

	// this can not be folded but load can be reused
	rect.max.x = rect.min.x * rect.min.x

	// this should invalidate the previous loads
	clobber(ptr)
	// now all stores are clobbered
	clobber(&rect.max)

	// conslusion: pointers fundamentally dont do anything and are not registered anywhere,
	// thay are just bound to the base memory and when you interact with them (store, load)
	// they modity the memory state, they are literally a view trought which we look at the
	// memory and remotely modify it, so in summary, pointers are not bound to a specific load
	// or store, but they can invalidate them.
	//
	// The fact pointers are bound to the base memory also makes it easy to tell how aliasing works
	// for the pointer, we prohibit pointer arithmetic on these pointers, instead this is delegated
	// to special pointer type that can only be created when compiler can prove ist safe or explicitly
	// with a directive

	return
}

clobber := fn(p: ^Point): void {
	*p = .(5, 6)
	return
}
