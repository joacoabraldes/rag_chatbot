// Settings panel — theme picker, source toggle, debug toggle, clear chat.

import { closeTrap, openTrap } from './a11y.js';
import { state, setDebugEnabled, getDebugEnabled } from './state.js';

const SHOW_SOURCES_KEY = 'showSources';

export function toggleSettings(show) {
    const overlay = document.getElementById('settings-overlay');
    const panel = document.getElementById('settings-panel');
    if (show) {
        overlay.classList.add('visible');
        panel.classList.add('visible');
        panel.setAttribute('aria-hidden', 'false');
        openTrap(panel, { initialFocus: document.getElementById('btn-close-settings') });
    } else {
        overlay.classList.remove('visible');
        panel.classList.remove('visible');
        panel.setAttribute('aria-hidden', 'true');
        closeTrap(panel);
    }
}

function initSourcesToggle() {
    const toggle = document.getElementById('toggle-sources');
    const stored = localStorage.getItem(SHOW_SOURCES_KEY);
    if (stored !== null) toggle.checked = stored !== 'false';
    toggle.addEventListener('change', () => {
        localStorage.setItem(SHOW_SOURCES_KEY, toggle.checked);
    });
}

function initDebugToggle() {
    const toggle = document.getElementById('toggle-debug');
    if (!toggle) return;
    toggle.checked = getDebugEnabled();
    applyDebugVisibility(toggle.checked);
    toggle.addEventListener('change', () => {
        setDebugEnabled(toggle.checked);
        applyDebugVisibility(toggle.checked);
    });
}

function applyDebugVisibility(enabled) {
    document.body.classList.toggle('show-debug', !!enabled);
}

function initClearChat(onClear) {
    document.getElementById('btn-clear-chat').addEventListener('click', () => {
        if (
            state.messages.length > 0
            && !confirm('Se borrará todo el historial de chat. ¿Continuar?')
        ) {
            return;
        }
        onClear();
        toggleSettings(false);
    });
}

export function initSettings({ onClear }) {
    document.getElementById('btn-settings')
        .addEventListener('click', () => toggleSettings(true));
    document.getElementById('btn-close-settings')
        .addEventListener('click', () => toggleSettings(false));
    document.getElementById('settings-overlay')
        .addEventListener('click', () => toggleSettings(false));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') toggleSettings(false);
    });

    initSourcesToggle();
    initDebugToggle();
    initClearChat(onClear);
}
