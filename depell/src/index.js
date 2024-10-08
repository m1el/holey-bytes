//// @ts-check

if (window.location.hostname === 'localhost') {
	let id; setInterval(async () => {
		let new_id = await fetch('/hot-reload').then(reps => reps.text());
		id ??= new_id;
		if (id !== new_id) window.location.reload();
	}, 300);
}

document.body.addEventListener('htmx:afterSwap', (ev) => {
	wireUp(ev.target);
});

wireUp(document.body);

/** @param {HTMLElement} target */
function wireUp(target) {
	execApply(target);
	cacheInputs(target);
	bindTextareaAutoResize(target);
}

/** @param {string} content @return {string} */
function fmtTimestamp(content) {
	new Date(parseInt(content) * 1000).toLocaleString()
}

/** @param {HTMLElement} target */
function execApply(target) {
	/**@type {HTMLElement}*/ let elem;
	for (elem of target.querySelectorAll('[apply]')) {
		const funcname = elem.getAttribute('apply');
		elem.textContent = window[funcname](elem.textContent);
	}
}

/** @param {HTMLElement} target */
function bindTextareaAutoResize(target) {
	/**@type {HTMLTextAreaElement}*/ let textarea;
	for (textarea of target.querySelectorAll("textarea")) {
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

		/**@type {HTMLInputElement}*/ let input;
		for (input of form.elements) {
			if ('password submit button'.includes(input.type)) continue;
			const key = path + input.name;
			input.value = localStorage.getItem(key) ?? '';
			input.addEventListener("input", (ev) => localStorage.setItem(key, ev.target.value));
		}
	}
}
