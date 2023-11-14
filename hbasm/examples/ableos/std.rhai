fn ipc_send(buffer_id, mem_addr, length){
    // set the ecall
    li8(r1, 3);
    // Set the buffer ID to be the BufferID
    li64(r2, buffer_id);
    lra(r3, r0, mem_addr);
    // set the length
    li64(r4, length);
    // ecall
    eca(); 
}

private fn log(log_level, string){
    let str = data::str(string);
    ipc_send(1, str, str.len);
}

fn Error(string) {log(0, string);}
fn Warn(string) {log(1, string);}
fn Info(string) {log(2, string);}
// Due to rhai limitations this cannot be debug
// because of this all of the log levels are upper case
fn Debug(string) {log(3, string);}
fn Trace(string) {log(4, string);}
