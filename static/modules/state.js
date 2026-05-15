// Shared mutable state. Kept tiny on purpose — anything heavier should
// live in a dedicated module.

export const state = {
    messages: [],
    isStreaming: false,
};

const SHOW_SOURCES_KEY = 'showSources';
const DEBUG_KEY = 'rag-chatbot-debug';

export function getShowSources() {
    const toggle = document.getElementById('toggle-sources');
    if (toggle) return toggle.checked;
    // Fallback to storage when DOM hasn't initialized yet.
    return localStorage.getItem(SHOW_SOURCES_KEY) !== 'false';
}

export function getDebugEnabled() {
    return localStorage.getItem(DEBUG_KEY) === 'true';
}

export function setDebugEnabled(enabled) {
    localStorage.setItem(DEBUG_KEY, enabled ? 'true' : 'false');
}
