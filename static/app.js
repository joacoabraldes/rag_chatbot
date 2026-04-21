// ─── State ─────────────────────────────────────────────────────────
const state = {
    messages: [],
    isStreaming: false,
};

const chatArea = document.getElementById('chat-area');
const welcomeScreen = document.getElementById('welcome-screen');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('btn-send');

// ─── Theme Management ──────────────────────────────────────────────
const THEME_KEY = 'rag-chatbot-theme';

function getStoredTheme() {
    return localStorage.getItem(THEME_KEY) || 'system';
}

function applyTheme(mode) {
    document.documentElement.dataset.theme = mode;
    localStorage.setItem(THEME_KEY, mode);
    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.themeVal === mode);
    });
}

if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (getStoredTheme() === 'system') {
            document.documentElement.dataset.theme = 'system';
        }
    });
}

// ─── Utilities ─────────────────────────────────────────────────────
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

function hideWelcomeScreen() {
    welcomeScreen.classList.add('hidden');
}

function showWelcomeScreen() {
    welcomeScreen.classList.remove('hidden');
}

function getShowSources() {
    const toggle = document.getElementById('toggle-sources');
    return toggle ? toggle.checked : true;
}

// ─── Render Messages ───────────────────────────────────────────────
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

function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="bubble typing-indicator">
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
        </div>`;
    chatArea.appendChild(div);
    scrollToBottom();
    return div;
}

function renderSourceBadges(bubble, sources) {
    if (!sources || sources.length === 0) return;

    const container = document.createElement('div');
    container.className = 'sources-list';

    sources.forEach((src) => {
        const ref = document.createElement('div');
        ref.className = 'source-ref';

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

        container.appendChild(ref);
    });

    bubble.appendChild(container);
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
    badge.textContent = label;
    bubble.appendChild(badge);
}

// ─── Send Message (streaming) ──────────────────────────────────────
async function sendMessage(query) {
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
        query: query,
        show_sources: showSources,
        history: history.length > 1 ? history.slice(0, -1) : null,
    };

    const typingEl = showTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        // Typing indicator stays until the first real token arrives (not just
        // when headers land) so the user does not see an empty bubble during
        // reasoning-model think time.
        let bubbleRefs = null;
        let firstToken = true;

        function ensureBubble() {
            if (!bubbleRefs) {
                typingEl.remove();
                bubbleRefs = createAssistantBubble();
            }
            return bubbleRefs;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = '';
        let sources = [];
        let meta = null;
        let buffer = '';
        let done = false;

        // Throttled rendering
        let renderTimer = null;
        let needsRender = false;

        function doRender(final) {
            if (!bubbleRefs) return;
            const { content } = bubbleRefs;
            const renderText = showSources
                ? fullAnswer
                : fullAnswer.replace(/\(Fecha:[^)]*\)\s*\[\d+\]/g, '').replace(/\[\d+\]/g, '');
            content.innerHTML = marked.parse(renderText);
            needsRender = false;
            if (final && renderTimer) {
                clearTimeout(renderTimer);
                renderTimer = null;
            }
            scrollToBottom();
        }

        function scheduleRender() {
            needsRender = true;
            if (!renderTimer) {
                renderTimer = setTimeout(() => {
                    renderTimer = null;
                    if (needsRender) doRender(false);
                }, 33);
            }
        }

        while (!done) {
            const { done: readerDone, value } = await reader.read();
            if (readerDone) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6);

                if (payload === '[END]') {
                    done = true;
                    break;
                }
                if (payload.startsWith('[ERROR]')) {
                    ensureBubble();
                    fullAnswer += payload.slice(7).trim();
                    doRender(true);
                    continue;
                }
                if (payload === '[DONE]') {
                    doRender(true);
                    continue;
                }
                if (payload.startsWith('[META] ')) {
                    try { meta = JSON.parse(payload.slice(7)); } catch(e) {}
                    const { bubble } = ensureBubble();
                    renderDebugBadge(bubble, meta);
                    continue;
                }
                if (payload.startsWith('[SOURCES] ')) {
                    try { sources = JSON.parse(payload.slice(10)); } catch(e) {}
                    const { bubble } = ensureBubble();
                    renderSourceBadges(bubble, sources);
                    scrollToBottom();
                    continue;
                }

                let token = payload;
                try { token = JSON.parse(payload); } catch (e) {}
                if (firstToken) {
                    ensureBubble();
                    firstToken = false;
                }
                fullAnswer += token;
                scheduleRender();
            }
        }

        if (needsRender || renderTimer) {
            doRender(true);
        }

        // Safety net: stream closed without any token, META, or SOURCES payload.
        if (!bubbleRefs) {
            const { content } = ensureBubble();
            content.textContent = 'Sin respuesta del modelo.';
        }

        state.messages.push({
            role: 'assistant',
            content: fullAnswer,
        });

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

// ─── Input Handling ────────────────────────────────────────────────
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

// ─── Settings Panel ────────────────────────────────────────────────
function toggleSettings(show) {
    const overlay = document.getElementById('settings-overlay');
    const panel = document.getElementById('settings-panel');
    if (show) {
        overlay.classList.add('visible');
        panel.classList.add('visible');
    } else {
        overlay.classList.remove('visible');
        panel.classList.remove('visible');
    }
}

document.getElementById('btn-settings').addEventListener('click', () => toggleSettings(true));
document.getElementById('btn-close-settings').addEventListener('click', () => toggleSettings(false));
document.getElementById('settings-overlay').addEventListener('click', () => toggleSettings(false));

// Theme buttons
document.querySelectorAll('.theme-option').forEach(btn => {
    btn.addEventListener('click', () => applyTheme(btn.dataset.themeVal));
});

// Clear chat
document.getElementById('btn-clear-chat').addEventListener('click', () => {
    if (state.messages.length > 0 && !confirm('Se borrará todo el historial de chat. ¿Continuar?')) {
        return;
    }
    state.messages = [];
    chatArea.querySelectorAll('.message').forEach(m => m.remove());
    showWelcomeScreen();
    toggleSettings(false);
});

// ─── Sources Toggle Persistence ─────────────────────────────────────
function initSourcesToggle() {
    const toggle = document.getElementById('toggle-sources');
    const stored = localStorage.getItem('showSources');
    if (stored !== null) toggle.checked = stored !== 'false';
    toggle.addEventListener('change', () => {
        localStorage.setItem('showSources', toggle.checked);
    });
}

// ─── Mobile Keyboard Handling ──────────────────────────────────────
if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', () => {
        setTimeout(() => scrollToBottom(), 100);
    });
}

// ─── Init ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    applyTheme(getStoredTheme());
    initSourcesToggle();
    userInput.focus();
});
