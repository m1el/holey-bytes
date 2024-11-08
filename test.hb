signum := fn($Expr: type, x: Expr): int {
    if x > @as(Expr, @intcast(0)) {
        return 1
    } else if x < @as(Expr, @intcast(0)) {
        return -
    } else {
        return 0
    }
}
