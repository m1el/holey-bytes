/// @ts-check

/** @return {never} */
function never() { throw new Error() }

/**@type{WebAssembly.Instance}*/ let hbcInstance;
/**@type{Promise<WebAssembly.WebAssemblyInstantiatedSource>}*/ let hbcInstaceFuture;
/** @param {Uint8Array} code @param {number} fuel
 * @returns {Promise<string | undefined> | string | undefined} */
function compileCode(code, fuel) {
	if (!hbcInstance) {
		hbcInstaceFuture ??= WebAssembly.instantiateStreaming(fetch("/hbc.wasm"), {});
		return (async () => {
			hbcInstance = (await hbcInstaceFuture).instance;
			return compileCodeSync(hbcInstance, code, fuel);
		})();
	} else {
		return compileCodeSync(hbcInstance, code, fuel);
	}

}

/** @param {WebAssembly.Instance} instance @param {Uint8Array} code @param {number} fuel @returns {string | undefined} */
function compileCodeSync(instance, code, fuel) {
	let {
		INPUT, INPUT_LEN,
		LOG_MESSAGES, LOG_MESSAGES_LEN,
		PANIC_MESSAGE, PANIC_MESSAGE_LEN,
		memory, compile_and_run
	} = instance.exports;

	if (!(true
		&& INPUT instanceof WebAssembly.Global
		&& INPUT_LEN instanceof WebAssembly.Global
		&& LOG_MESSAGES instanceof WebAssembly.Global
		&& LOG_MESSAGES_LEN instanceof WebAssembly.Global
		&& memory instanceof WebAssembly.Memory
		&& typeof compile_and_run === "function"
	)) console.log(instance.exports), never();

	const dw = new DataView(memory.buffer);
	dw.setUint32(INPUT_LEN.value, code.length, true);
	new Uint8Array(memory.buffer, INPUT.value).set(code);

	try {
		compile_and_run(fuel);
		return bufToString(memory, LOG_MESSAGES, LOG_MESSAGES_LEN);
	} catch (e) {
		if (PANIC_MESSAGE instanceof WebAssembly.Global
			&& PANIC_MESSAGE_LEN instanceof WebAssembly.Global) {
			console.error(bufToString(memory, PANIC_MESSAGE, PANIC_MESSAGE_LEN));
		}
		let log = bufToString(memory, LOG_MESSAGES, LOG_MESSAGES_LEN);
		console.error(log, e);
		return undefined;
	}
}

/** @param {WebAssembly.Memory} mem
 * @param {WebAssembly.Global} ptr
 * @param {WebAssembly.Global} len
 * @return {string} */
function bufToString(mem, ptr, len) {
	return new TextDecoder()
		.decode(new Uint8Array(mem.buffer, ptr.value,
			new DataView(mem.buffer).getUint32(len.value, true)));
}

/**@type{WebAssembly.Instance}*/ let fmtInstance;
/**@type{Promise<WebAssembly.WebAssemblyInstantiatedSource>}*/ let fmtInstaceFuture;
/** @param {string} code @param {"fmt" | "minify"} action
 * @returns {Promise<string | undefined> | string | undefined} */
function modifyCode(code, action) {
	if (!fmtInstance) {
		fmtInstaceFuture ??= WebAssembly.instantiateStreaming(fetch("/hbfmt.wasm"), {});
		return (async () => {
			fmtInstance = (await fmtInstaceFuture).instance;
			return modifyCodeSync(fmtInstance, code, action);
		})();
	} else {
		return modifyCodeSync(fmtInstance, code, action);
	}
}

/** @param {WebAssembly.Instance} instance @param {string} code @param {"fmt" | "minify"} action @returns {string | undefined} */
function modifyCodeSync(instance, code, action) {
	let {
		INPUT, INPUT_LEN,
		OUTPUT, OUTPUT_LEN,
		PANIC_MESSAGE, PANIC_MESSAGE_LEN,
		memory, fmt, minify
	} = instance.exports;

	if (!(true
		&& INPUT instanceof WebAssembly.Global
		&& INPUT_LEN instanceof WebAssembly.Global
		&& OUTPUT instanceof WebAssembly.Global
		&& OUTPUT_LEN instanceof WebAssembly.Global
		&& memory instanceof WebAssembly.Memory
		&& typeof fmt === "function"
		&& typeof minify === "function"
	)) never();

	if (action !== "fmt") {
		INPUT = OUTPUT;
		INPUT_LEN = OUTPUT_LEN;
	}

	let dw = new DataView(memory.buffer);
	dw.setUint32(INPUT_LEN.value, code.length, true);
	new Uint8Array(memory.buffer, INPUT.value)
		.set(new TextEncoder().encode(code));

	try {
		if (action === "fmt") fmt(); else minify();
		let result = new TextDecoder()
			.decode(new Uint8Array(memory.buffer, OUTPUT.value,
				dw.getUint32(OUTPUT_LEN.value, true)));
		return result;
	} catch (e) {
		if (PANIC_MESSAGE instanceof WebAssembly.Global
			&& PANIC_MESSAGE_LEN instanceof WebAssembly.Global) {
			let message = new TextDecoder()
				.decode(new Uint8Array(memory.buffer, PANIC_MESSAGE.value,
					dw.getUint32(PANIC_MESSAGE_LEN.value, true)));
			console.error(message, e);
		} else {
			console.error(e);
		}
		return undefined;
	}
}

/** @param {HTMLElement} target */
function wireUp(target) {
	execApply(target);
	cacheInputs(target);
	bindTextareaAutoResize(target);
}

/** @type {{ [key: string]: (content: string) => Promise<string> | string }} */
const applyFns = {
	timestamp: (content) => new Date(parseInt(content) * 1000).toLocaleString(),
	fmt: (content) => {
		let res = modifyCode(content, "fmt");
		return res instanceof Promise ? res.then(c => c ?? content) : res ?? content;
	},
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
			textarea.style.height = "auto";
			textarea.style.height = (textarea.scrollHeight - padding) + "px";
		});

		textarea.onkeydown = (ev) => {
			const selecting = textarea.selectionStart !== textarea.selectionEnd;

			if (ev.key === "Tab") {
				ev.preventDefault();
				document.execCommand('insertText', false, "    ");
			}

			if (ev.key === "Backspace" && textarea.selectionStart != 0 && !selecting) {
				let i = textarea.selectionStart, looped = false;
				while (textarea.value.charCodeAt(--i) === ' '.charCodeAt(0)) looped = true;
				if (textarea.value.charCodeAt(i) === '\n'.charCodeAt(0) && looped) {
					ev.preventDefault();
					let toDelete = (textarea.selectionStart - (i + 1)) % 4;
					if (toDelete === 0) toDelete = 4;
					textarea.selectionStart -= toDelete;
					document.execCommand('delete', false);
				}
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
			const fmtd = await modifyCode(code, "fmt") ?? never();
			const prev = await modifyCode(fmtd, "minify") ?? never();
			if (code != prev) console.error(code, prev);
		}
		{

			const name = "foo.hb";
			const code = "main:=fn():void{return 42}";
			const buf = new Uint8Array(2 + name.length + 2 + code.length);
			const view = new DataView(buf.buffer);
			view.setUint16(0, name.length, true);
			buf.set(new TextEncoder().encode(name), 2);
			view.setUint16(2 + name.length, code.length, true);
			buf.set(new TextEncoder().encode(code), 2 + name.length + 2);
			const res = await compileCode(buf, 1) ?? never();
			const expected = "";
			if (expected != res) console.error(expected, res);
		}
	})()
}

document.body.addEventListener('htmx:afterSwap', (ev) => {
	if (!(ev.target instanceof HTMLElement)) never();
	wireUp(ev.target);
});

wireUp(document.body);
