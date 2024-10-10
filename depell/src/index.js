/// @ts-check

/** @return {never} */
function never() { throw new Error() }

/**@type{WebAssembly.Instance}*/ let instance;
/**@type{Promise<WebAssembly.WebAssemblyInstantiatedSource>}*/ let instaceFuture;
/** @param {string} code @param {"fmt" | "minify"} action
 * @returns {Promise<string | undefined> | string | undefined} */
function modifyCode(code, action) {
	if (!instance) {
		instaceFuture ??= WebAssembly.instantiateStreaming(fetch("/hbfmt.wasm"), {});
		return (async () => {
			instance = (await instaceFuture).instance;
			return modifyCodeSync(instance, code, action);
		})();
	} else {
		return modifyCodeSync(instance, code, action);
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

		textarea.style.height = textarea.scrollHeight + "px";
		textarea.style.overflowY = "hidden";
		textarea.addEventListener("input", function() {
			textarea.style.height = "auto";
			textarea.style.height = textarea.scrollHeight + "px";
		});

		textarea.onkeydown = (ev) => {
			const selecting = textarea.selectionStart !== textarea.selectionEnd;

			if (ev.key === "Tab") {
				ev.preventDefault();
				const prevPos = textarea.selectionStart;
				textarea.value = textarea.value.slice(0, textarea.selectionStart) +
					'    ' + textarea.value.slice(textarea.selectionEnd);
				textarea.selectionStart = textarea.selectionEnd = prevPos + 4;
			}

			if (ev.key === "Backspace" && textarea.selectionStart != 0 && !selecting) {
				let i = textarea.selectionStart, looped = false;
				while (textarea.value.charCodeAt(--i) === ' '.charCodeAt(0)) looped = true;
				if (textarea.value.charCodeAt(i) === '\n'.charCodeAt(0) && looped) {
					ev.preventDefault();
					let toDelete = (textarea.selectionStart - (i + 1)) % 4;
					if (toDelete === 0) toDelete = 4;
					const prevPos = textarea.selectionStart;
					textarea.value = textarea.value.slice(0, textarea.selectionStart - toDelete) +
						textarea.value.slice(textarea.selectionEnd);
					textarea.selectionStart = textarea.selectionEnd = prevPos - toDelete;
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

	(async function testCodeChange() {
		const code = "main:=fn():void{return}";
		const fmtd = await modifyCode(code, "fmt") ?? never();
		const prev = await modifyCode(fmtd, "minify") ?? never();
		if (code != prev) console.error(code, prev);
	})()
}

document.body.addEventListener('htmx:afterSwap', (ev) => {
	if (!(ev.target instanceof HTMLElement)) never();
	wireUp(ev.target);
});

wireUp(document.body);
