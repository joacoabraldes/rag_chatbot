// Tiny focus-trap helper for modal panels (settings, source drawer).
// On open: save the currently focused element, move focus inside, trap Tab.
// On close: restore the original focus.

const TRAPS = new Map();

const FOCUSABLE = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
].join(',');

export function openTrap(container, { initialFocus } = {}) {
    if (!container || TRAPS.has(container)) return;

    const previouslyFocused = document.activeElement;
    const handler = (e) => {
        if (e.key !== 'Tab') return;
        const items = Array.from(container.querySelectorAll(FOCUSABLE))
            .filter(el => el.offsetParent !== null);
        if (items.length === 0) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
            e.preventDefault();
            last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault();
            first.focus();
        }
    };
    container.addEventListener('keydown', handler);
    TRAPS.set(container, { handler, previouslyFocused });

    // Move focus inside. Prefer the requested initial target, fall back to
    // the first focusable element.
    const target = initialFocus
        || container.querySelector(FOCUSABLE);
    if (target && typeof target.focus === 'function') {
        // setTimeout 0 lets the panel finish its transition before focus moves.
        setTimeout(() => target.focus({ preventScroll: true }), 0);
    }
}

export function closeTrap(container) {
    const trap = TRAPS.get(container);
    if (!trap) return;
    container.removeEventListener('keydown', trap.handler);
    TRAPS.delete(container);
    const prev = trap.previouslyFocused;
    if (prev && typeof prev.focus === 'function') {
        try { prev.focus({ preventScroll: true }); } catch (_) {}
    }
}
