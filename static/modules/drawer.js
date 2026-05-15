// Source preview drawer — opens on citation click, shows chunk text + metadata.

import { closeTrap, openTrap } from './a11y.js';

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

export function openSourceDrawer(src) {
    const drawer = document.getElementById('source-drawer');
    const overlay = document.getElementById('source-overlay');
    document.getElementById('source-drawer-idx').textContent = `[${src.index}]`;
    document.getElementById('source-drawer-name').textContent =
        src.source_file || 'desconocido';

    const meta = document.getElementById('source-drawer-meta');
    meta.innerHTML = '';
    if (src.pub_date) {
        const el = document.createElement('span');
        el.innerHTML = `<strong>Fecha:</strong> ${escapeHtml(src.pub_date)}`;
        meta.appendChild(el);
    }
    if (src.page_number) {
        const el = document.createElement('span');
        el.innerHTML = `<strong>Página:</strong> ${escapeHtml(String(src.page_number))}`;
        meta.appendChild(el);
    }
    if (src.topic_tags) {
        const el = document.createElement('span');
        el.innerHTML = `<strong>Temas:</strong> ${escapeHtml(src.topic_tags)}`;
        meta.appendChild(el);
    }

    const textEl = document.getElementById('source-drawer-text');
    textEl.textContent = src.text || 'Sin texto disponible.';

    overlay.classList.add('visible');
    drawer.classList.add('visible');
    drawer.setAttribute('aria-hidden', 'false');

    openTrap(drawer, { initialFocus: document.getElementById('btn-close-source') });
}

export function closeSourceDrawer() {
    const drawer = document.getElementById('source-drawer');
    const overlay = document.getElementById('source-overlay');
    if (!drawer.classList.contains('visible')) return;
    drawer.classList.remove('visible');
    overlay.classList.remove('visible');
    drawer.setAttribute('aria-hidden', 'true');
    closeTrap(drawer);
}

export function initDrawer() {
    document.getElementById('btn-close-source').addEventListener('click', closeSourceDrawer);
    document.getElementById('source-overlay').addEventListener('click', closeSourceDrawer);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeSourceDrawer();
    });
}
