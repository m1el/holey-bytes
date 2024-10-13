/// @ts-check

/** @return {never} */
function never() { throw new Error() }

/**@type{WebAssembly.Instance}*/ let hbcInstance;
/**@type{Promise<WebAssembly.WebAssemblyInstantiatedSource>}*/ let hbcInstaceFuture;
async function getHbcInstance() {
	hbcInstaceFuture ??= WebAssembly.instantiateStreaming(fetch("/hbc.wasm"), {});
	return hbcInstance ??= (await hbcInstaceFuture).instance;
}

const stack_pointer_offset = 1 << 20;

/** @param {WebAssembly.Instance} instance @param {Post[]} packages @param {number} fuel
 * @returns {string} */
function compileCode(instance, packages, fuel) {
	let {
		INPUT, INPUT_LEN,
		LOG_MESSAGES, LOG_MESSAGES_LEN,
		memory, compile_and_run,
	} = instance.exports;

	if (!(true
		&& memory instanceof WebAssembly.Memory
		&& INPUT instanceof WebAssembly.Global
		&& INPUT_LEN instanceof WebAssembly.Global
		&& LOG_MESSAGES instanceof WebAssembly.Global
		&& LOG_MESSAGES_LEN instanceof WebAssembly.Global
		&& typeof compile_and_run === "function"
	)) never();

	const codeLength = packPosts(packages, new DataView(memory.buffer, INPUT.value));
	new DataView(memory.buffer).setUint32(INPUT_LEN.value, codeLength, true);

	runWasmFunction(instance, compile_and_run, fuel);
	return bufToString(memory, LOG_MESSAGES, LOG_MESSAGES_LEN);
}

/**@type{WebAssembly.Instance}*/ let fmtInstance;
/**@type{Promise<WebAssembly.WebAssemblyInstantiatedSource>}*/ let fmtInstaceFuture;
async function getFmtInstance() {
	fmtInstaceFuture ??= WebAssembly.instantiateStreaming(fetch("/hbfmt.wasm"), {});
	return fmtInstance ??= (await fmtInstaceFuture).instance;
}

/** @param {WebAssembly.Instance} instance @param {string} code @param {"fmt" | "minify"} action
 * @returns {string | undefined} */
function modifyCode(instance, code, action) {
	let {
		INPUT, INPUT_LEN,
		OUTPUT, OUTPUT_LEN,
		memory, fmt, minify
	} = instance.exports;

	if (!(true
		&& memory instanceof WebAssembly.Memory
		&& INPUT instanceof WebAssembly.Global
		&& INPUT_LEN instanceof WebAssembly.Global
		&& OUTPUT instanceof WebAssembly.Global
		&& OUTPUT_LEN instanceof WebAssembly.Global
		&& typeof fmt === "function"
		&& typeof minify === "function"
	)) never();

	if (action !== "fmt") {
		INPUT = OUTPUT;
		INPUT_LEN = OUTPUT_LEN;
	}

	let dw = new DataView(memory.buffer);
	dw.setUint32(INPUT_LEN.value, code.length, true);
	new Uint8Array(memory.buffer, INPUT.value).set(new TextEncoder().encode(code));

	return runWasmFunction(instance, action === "fmt" ? fmt : minify) ?
		bufToString(memory, OUTPUT, OUTPUT_LEN) : undefined;
}


/** @param {WebAssembly.Instance} instance @param {CallableFunction} func @param {any[]} args
 * @returns {boolean} */
function runWasmFunction(instance, func, ...args) {
	//const prev = performance.now();
	const { PANIC_MESSAGE, PANIC_MESSAGE_LEN, memory, stack_pointer } = instance.exports;
	if (!(true
		&& memory instanceof WebAssembly.Memory
		&& stack_pointer instanceof WebAssembly.Global
	)) never();
	const ptr = stack_pointer.value;
	try {
		func(...args);
		return true;
	} catch (error) {
		if (error instanceof WebAssembly.RuntimeError && error.message == "unreachable") {
			if (PANIC_MESSAGE instanceof WebAssembly.Global
				&& PANIC_MESSAGE_LEN instanceof WebAssembly.Global) {
				console.error(bufToString(memory, PANIC_MESSAGE, PANIC_MESSAGE_LEN), error);
			}
		} else {
			console.error(error);
		}
		stack_pointer.value = ptr;
		return false;
	} finally {
		//console.log("compiletion took:", performance.now() - prev);
	}
}

/** @typedef {Object} Post
 * @property {string} path 
 * @property {string} code */

/** @param {Post[]} posts @param {DataView} view  @returns {number} */
function packPosts(posts, view) {
	const enc = new TextEncoder(), buf = new Uint8Array(view.buffer, view.byteOffset);
	let len = 0; for (const post of posts) {
		view.setUint16(len, post.path.length, true); len += 2;
		buf.set(enc.encode(post.path), len); len += post.path.length;
		view.setUint16(len, post.code.length, true); len += 2;
		buf.set(enc.encode(post.code), len); len += post.code.length;
	}
	return len;
}

/** @param {WebAssembly.Memory} mem
 * @param {WebAssembly.Global} ptr
 * @param {WebAssembly.Global} len
 * @return {string} */
function bufToString(mem, ptr, len) {
	const res = new TextDecoder()
		.decode(new Uint8Array(mem.buffer, ptr.value,
			new DataView(mem.buffer).getUint32(len.value, true)));
	new DataView(mem.buffer).setUint32(len.value, 0, true);
	return res;
}

/** @param {HTMLElement} target */
function wireUp(target) {
	execApply(target);
	cacheInputs(target);
	bindTextareaAutoResize(target);
	bindCodeEdit(target);
}

const importRe = /@use\s*\(\s*"(([^"]|\\")+)"\s*\)/g;
/** @param {string} code @param {string[]} matches @returns {string[]} */
function findImports(code, matches) {
	matches.length = 0;
	for (const match of code.matchAll(importRe)) {
		matches.push(match[1]);
	}

	matches.sort();

	let c = 0;
	for (let i = 1; i < matches.length; i++) {
		if (matches[c] != matches[i]) {
			matches[++c] = matches[i];
		}
	}
	matches.length = Math.min(matches.length, c + 1);

	return matches;
}


/** @param {HTMLElement} target */
async function bindCodeEdit(target) {
	const edit = target.querySelector("#code-edit");
	if (!(edit instanceof HTMLTextAreaElement)) return;

	const codeSize = target.querySelector("#code-size");
	const errors = target.querySelector("#compiler-output");
	if (!(true
		&& codeSize instanceof HTMLSpanElement
		&& errors instanceof HTMLPreElement
	)) never();

	const MAX_CODE_SIZE = parseInt(codeSize.innerHTML);
	if (Number.isNaN(MAX_CODE_SIZE)) never();

	const hbc = await getHbcInstance(), fmt = await getFmtInstance();
	const prevImports = [];
	const matches = [];
	/**@type{Post[]}*/
	const packages = [{ path: "local.hb", code: "" }];
	const debounce = 100;
	/**@type{AbortController|undefined}*/
	let cancelation = undefined;
	let timeout = 0;

	edit.addEventListener("input", () => {
		if (timeout) clearTimeout(timeout);
		timeout = setTimeout(() => {
			prevImports.length = 0; prevImports.push(...matches);
			const imports = findImports(edit.value, matches);
			let changed = imports.length !== prevImports.length;
			for (let i = 0; i < imports.length && !changed; i++) {
				changed ||= imports[i] !== prevImports[i];
			}

			if (changed && imports.length !== 0) {
				if (cancelation) cancelation.abort();
				cancelation = new AbortController();
				errors.textContent = "fetching: " + imports.join(", ");
				fetch(`/code`, {
					method: "POST",
					signal: cancelation.signal,
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify(imports),
				}).then(async e => {
					try {
						const json = await e.json();
						if (e.status == 200) {
							packages.length = 1;
							packages.push(...json);
							cancelation = undefined;
							edit.dispatchEvent(new InputEvent("input"));
						} else {
							errors.textContent = "failed to fetch: " + json.join(", ");
						}
					} catch (er) {
						errors.textContent = "completely failed to fetch ("
							+ e.status + "): " + imports.join(", ");
						console.error(e);
					}
				});
			}

			if (cancelation && imports.length !== 0) {
				return;
			}

			packages[0].code = edit.value;

			errors.textContent = compileCode(hbc, packages, 1);
			const minified_size = modifyCode(fmt, edit.value, "minify")?.length;
			if (minified_size) {
				codeSize.textContent = (MAX_CODE_SIZE - minified_size) + "";
				const perc = Math.min(100, Math.floor(100 * (minified_size / MAX_CODE_SIZE)));
				codeSize.style.color = `color-mix(in srgb, white, var(--error) ${perc}%)`;
			}
			timeout = 0;
		}, debounce);
	});
	edit.dispatchEvent(new InputEvent("input"));
}

/** @type {{ [key: string]: (content: string) => Promise<string> | string }} */
const applyFns = {
	timestamp: (content) => new Date(parseInt(content) * 1000).toLocaleString(),
	fmt: (content) => getFmtInstance().then(i => modifyCode(i, content, "fmt") ?? "invalid code"),
};
/** @param {HTMLElement} target */
function execApply(target) {
	for (const elem of target.querySelectorAll('[apply]')) {
		if (!(elem instanceof HTMLElement)) continue;
		const funcname = elem.getAttribute('apply') ?? never();
		let res = applyFns[funcname](elem.textContent ?? "");
		if (res instanceof Promise) res.then(c => elem.textContent = c);
		else elem.textContent = res;
	}
}

/** @param {HTMLElement} target */
function bindTextareaAutoResize(target) {
	for (const textarea of target.querySelectorAll("textarea")) {
		if (!(textarea instanceof HTMLTextAreaElement)) never();

		const taCssMap = window.getComputedStyle(textarea);
		const padding = parseInt(taCssMap.getPropertyValue('padding-top') ?? "0")
			+ parseInt(taCssMap.getPropertyValue('padding-top') ?? "0");
		textarea.style.height = "auto";
		textarea.style.height = (textarea.scrollHeight - padding) + "px";
		textarea.style.overflowY = "hidden";
		textarea.addEventListener("input", function() {
			let top = window.scrollY;
			textarea.style.height = "auto";
			textarea.style.height = (textarea.scrollHeight - padding) + "px";
			window.scrollTo({ top });
		});

		textarea.onkeydown = (ev) => {
			if (ev.key === "Tab") {
				ev.preventDefault();
				document.execCommand('insertText', false, "\t");
			}
		}
	}
}

/** @param {HTMLElement} target */
function cacheInputs(target) {
	/**@type {HTMLFormElement}*/ let form;
	for (form of target.querySelectorAll('form')) {
		const path = form.getAttribute('hx-post') || form.getAttribute('hx-delete');
		if (!path) {
			console.warn('form does not have a hx-post or hx-delete attribute', form);
			continue;
		}

		for (const input of form.elements) {
			if (input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement) {
				if ('password submit button'.includes(input.type)) continue;
				const key = path + input.name;
				input.value = localStorage.getItem(key) ?? '';
				input.addEventListener("input", () => localStorage.setItem(key, input.value));
			} else {
				console.warn("unhandled form element: ", input);
			}
		}
	}
}

if (window.location.hostname === 'localhost') {
	let id; setInterval(async () => {
		let new_id = await fetch('/hot-reload').then(reps => reps.text());
		id ??= new_id;
		if (id !== new_id) window.location.reload();
	}, 300);

	(async function test() {
		{
			const code = "main:=fn():void{return}";
			const inst = await getFmtInstance()
			const fmtd = modifyCode(inst, code, "fmt") ?? never();
			const prev = modifyCode(inst, fmtd, "minify") ?? never();
			if (code != prev) console.error(code, prev);
		}
		{
			const posts = [{
				path: "foo.hb",
				code: "main:=fn():int{return 42}",
			}];
			const res = compileCode(await getHbcInstance(), posts, 1) ?? never();
			const expected = "exit code: 42\n";
			if (expected != res) console.error(expected, res);
		}
	})()
}

document.body.addEventListener('htmx:afterSwap', (ev) => {
	if (!(ev.target instanceof HTMLElement)) never();
	wireUp(ev.target);
});

wireUp(document.body);
