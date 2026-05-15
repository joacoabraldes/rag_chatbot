// Chat rendering + send pipeline. The "thick" module — handles message
// DOM, the streaming flow, and orchestrates stream events → UI updates.

import { openSourceDrawer } from './drawer.js';
import { IncrementalMarkdown } from './markdown.js';
import { state, getShowSources } from './state.js';
import { parseSSE } from './stream.js';

const chatArea = document.getElementById('chat-area');
const welcomeScreen = document.getElementById('welcome-screen');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('btn-send');

// ─── DOM helpers ──────────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

let _scrollPending = false;
function scrollToBottom() {
    if (_scrollPending) return;
    _scrollPending = true;
    requestAnimationFrame(() => {
        chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
        _scrollPending = false;
    });
}

export function hideWelcomeScreen() {
    welcomeScreen.classList.add('hidden');
}
export function showWelcomeScreen() {
    welcomeScreen.classList.remove('hidden');
}

export function clearMessages() {
    state.messages = [];
    chatArea.querySelectorAll('.message').forEach(m => m.remove());
    showWelcomeScreen();
}

// ─── Renderers ────────────────────────────────────────────────────

function renderUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chatArea.appendChild(div);
    scrollToBottom();
}

function createAssistantBubble() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    const content = document.createElement('div');
    content.className = 'message-content';
    bubble.appendChild(content);
    msgDiv.appendChild(bubble);
    chatArea.appendChild(msgDiv);
    return { msgDiv, bubble, content };
}

function showSkeleton() {
    // Lightweight skeleton lines instead of the old typing-bar indicator.
    // Communicates "I'm about to write" instead of "I'm thinking" — feels
    // less idle while the reasoning model is silent.
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="bubble skeleton-bubble">
            <div class="skeleton-line w90"></div>
            <div class="skeleton-line w75"></div>
            <div class="skeleton-line w60"></div>
        </div>`;
    chatArea.appendChild(div);
    scrollToBottom();
    return div;
}

function renderSourceBadges(bubble, sources) {
    if (!sources || sources.length === 0) return;

    const container = document.createElement('div');
    container.className = 'sources-list';
    container.setAttribute('role', 'list');

    sources.forEach((src) => {
        const ref = document.createElement('div');
        ref.className = 'source-ref';
        ref.setAttribute('role', 'listitem');
        ref.setAttribute('tabindex', '0');
        ref.title = 'Ver fragmento citado';

        const idxSpan = document.createElement('span');
        idxSpan.className = 'source-ref-idx';
        idxSpan.textContent = `[${src.index}]`;

        const nameSpan = document.createElement('span');
        nameSpan.className = 'source-ref-name';
        nameSpan.textContent = src.source_file || 'desconocido';

        ref.appendChild(idxSpan);
        ref.appendChild(nameSpan);

        if (src.pub_date) {
            const dateSpan = document.createElement('span');
            dateSpan.className = 'source-ref-date';
            dateSpan.textContent = src.pub_date;
            ref.appendChild(dateSpan);
        }

        if (src.page_number) {
            const pageSpan = document.createElement('span');
            pageSpan.className = 'source-ref-page';
            pageSpan.textContent = `p.${src.page_number}`;
            ref.appendChild(pageSpan);
        }

        ref.addEventListener('click', () => openSourceDrawer(src));
        ref.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openSourceDrawer(src);
            }
        });

        container.appendChild(ref);
    });

    bubble.appendChild(container);
}

function renderFollowups(bubble, followups) {
    if (!followups || followups.length === 0) return;

    const container = document.createElement('div');
    container.className = 'followups';

    const label = document.createElement('div');
    label.className = 'followups-label';
    label.innerHTML = `
        <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
        <span>Seguir explorando</span>
    `;
    container.appendChild(label);

    const chips = document.createElement('div');
    chips.className = 'followups-chips';
    chips.setAttribute('role', 'list');
    followups.forEach((q) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'followup-chip';
        btn.setAttribute('role', 'listitem');
        btn.textContent = q;
        btn.addEventListener('click', () => {
            if (state.isStreaming) return;
            container.querySelectorAll('.followup-chip').forEach(c => c.disabled = true);
            sendMessage(q);
        });
        chips.appendChild(btn);
    });
    container.appendChild(chips);

    bubble.appendChild(container);
    scrollToBottom();
}

function renderDebugBadge(bubble, meta) {
    if (!meta) return;
    const badge = document.createElement('div');
    badge.className = 'debug-badge';

    let label;
    if (!meta.retrieve) {
        label = 'modo directo';
        badge.dataset.mode = 'direct';
    } else if (meta.chunks === 0) {
        label = 'RAG · 0 chunks';
        badge.dataset.mode = 'rag-empty';
    } else if (meta.sources_sent) {
        label = `RAG · ${meta.chunks} chunks · fuentes ✓`;
        badge.dataset.mode = 'rag-sources';
    } else {
        label = `RAG · ${meta.chunks} chunks · fuentes ✗`;
        badge.dataset.mode = 'rag-nosources';
    }

    if (typeof meta.ttft_ms === 'number') {
        label += ` · ttft ${meta.ttft_ms}ms`;
    }
    if (typeof meta.n_invalid_citations === 'number' && meta.n_invalid_citations > 0) {
        label += ` · ⚠ ${meta.n_invalid_citations} cita(s) inválida(s)`;
    }
    badge.textContent = label;
    bubble.appendChild(badge);
}

// ─── Send / stream pipeline ───────────────────────────────────────

export async function sendMessage(query) {
    if (state.isStreaming || !query.trim()) return;
    state.isStreaming = true;
    sendBtn.disabled = true;
    userInput.value = '';
    userInput.style.height = 'auto';

    hideWelcomeScreen();

    state.messages.push({ role: 'user', content: query });
    renderUserMessage(query);

    const history = state.messages
        .slice(-10)
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content }));

    const showSources = getShowSources();

    const body = {
        query,
        show_sources: showSources,
        history: history.length > 1 ? history.slice(0, -1) : null,
    };

    const typingEl = showSkeleton();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        // Bubble lazily built on first content token, so the skeleton stays
        // visible during reasoning model think-time (no empty bubble flash).
        let bubbleRefs = null;
        let mdRenderer = null;
        let fullAnswer = '';
        let renderTimer = null;
        let needsRender = false;

        const ensureBubble = () => {
            if (!bubbleRefs) {
                typingEl.remove();
                bubbleRefs = createAssistantBubble();
                mdRenderer = new IncrementalMarkdown(bubbleRefs.content);
            }
            return bubbleRefs;
        };

        const renderNow = (final) => {
            if (!mdRenderer) return;
            // Strip RAG citation noise when sources panel is off.
            const renderText = showSources
                ? fullAnswer
                : fullAnswer.replace(/\(Fecha:[^)]*\)\s*\[\d+\]/g, '').replace(/\[\d+\]/g, '');
            if (final) {
                mdRenderer.finalize(renderText);
            } else {
                mdRenderer.update(renderText);
            }
            needsRender = false;
            if (final && renderTimer) {
                clearTimeout(renderTimer);
                renderTimer = null;
            }
            scrollToBottom();
        };

        const scheduleRender = () => {
            needsRender = true;
            if (!renderTimer) {
                renderTimer = setTimeout(() => {
                    renderTimer = null;
                    if (needsRender) renderNow(false);
                }, 33);
            }
        };

        for await (const ev of parseSSE(response)) {
            if (ev.type === 'token') {
                ensureBubble();
                fullAnswer += ev.text;
                scheduleRender();
            } else if (ev.type === 'done') {
                renderNow(true);
            } else if (ev.type === 'error') {
                ensureBubble();
                fullAnswer += ev.text;
                renderNow(true);
            } else if (ev.type === 'meta') {
                const { bubble } = ensureBubble();
                renderDebugBadge(bubble, ev.data);
            } else if (ev.type === 'sources') {
                const { bubble } = ensureBubble();
                renderSourceBadges(bubble, ev.data);
                scrollToBottom();
            } else if (ev.type === 'followups') {
                const { bubble } = ensureBubble();
                renderFollowups(bubble, ev.data);
            } else if (ev.type === 'end') {
                break;
            }
        }

        if (needsRender || renderTimer) renderNow(true);

        // Safety net: stream closed without producing anything visible.
        if (!bubbleRefs) {
            const { content } = ensureBubble();
            content.textContent = 'Sin respuesta del modelo.';
        }

        state.messages.push({ role: 'assistant', content: fullAnswer });
        scrollToBottom();
    } catch (e) {
        typingEl.remove();
        const { content } = createAssistantBubble();
        content.textContent = 'Ocurrió un error al procesar la consulta. Por favor, intentá de nuevo.';
        state.messages.push({ role: 'assistant', content: content.textContent });
        console.error('Stream error:', e);
    }

    state.isStreaming = false;
    sendBtn.disabled = !userInput.value.trim();
}

// ─── Input handling ───────────────────────────────────────────────

export function initInput() {
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
        sendBtn.disabled = !userInput.value.trim() || state.isStreaming;
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (userInput.value.trim() && !state.isStreaming) {
                sendMessage(userInput.value.trim());
            }
        }
    });

    sendBtn.addEventListener('click', () => {
        if (userInput.value.trim() && !state.isStreaming) {
            sendMessage(userInput.value.trim());
        }
    });

    // Mobile: keep the latest message visible when the on-screen keyboard
    // opens. visualViewport resize fires when keyboard slides in/out.
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', () => {
            setTimeout(scrollToBottom, 100);
        });
    }
}
