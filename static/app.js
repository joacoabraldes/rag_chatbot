// ─── State ─────────────────────────────────────────────────────────
const state = {
    messages: [],
    collections: [],
    selectedCollections: [],
    isStreaming: false,
};

const chatArea = document.getElementById('chat-area');
const welcomeScreen = document.getElementById('welcome-screen');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('btn-send');

// ─── Theme Management ──────────────────────────────────────────────
const THEME_KEY = 'rag-analyst-theme';

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

// ─── Dynamic top-k estimation ──────────────────────────────────────
function estimateK(query) {
    const words = query.trim().split(/\s+/);
    const wordCount = words.length;
    const lower = query.toLowerCase();

    let k;
    if (wordCount <= 4) k = 6;
    else if (wordCount <= 10) k = 10;
    else k = 13;

    const broadTerms = /\b(compar[aáeé]|diferencia|vs\.?|entre|relacion|analiz[aáeé]|contrasta)\b/i;
    if (broadTerms.test(lower)) k += 3;

    const enumTerms = /\b(cuales|cu[aá]les|lista|todos|principales|factores|resumen general)\b/i;
    if (enumTerms.test(lower)) k += 2;

    const specificIndicators = /\b(\d{4}|\d{1,2}[\/\-]\d{1,2}|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b/i;
    if (specificIndicators.test(lower)) k -= 2;

    const definitionTerms = /\b(que es|qué es|definición|definicion|significa)\b/i;
    if (definitionTerms.test(lower)) k -= 2;

    return Math.max(5, Math.min(15, k));
}

// ─── Collections ───────────────────────────────────────────────────
async function loadCollections() {
    try {
        const res = await fetch('/collections');
        const data = await res.json();
        state.collections = data.collections || [];
        state.selectedCollections = [...state.collections];
        renderCollectionCheckboxes();
    } catch (e) {
        console.error('Failed to load collections:', e);
    }
}

function renderCollectionCheckboxes() {
    const container = document.getElementById('collection-list');
    container.innerHTML = '';
    if (state.collections.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;padding:8px 12px">No hay colecciones disponibles.</p>';
        return;
    }
    for (const col of state.collections) {
        const item = document.createElement('div');
        item.className = 'collection-item';
        const checked = state.selectedCollections.includes(col) ? 'checked' : '';
        item.innerHTML = `<input type="checkbox" id="col-${col}" ${checked}><label for="col-${col}">${escapeHtml(col)}</label>`;
        const cb = item.querySelector('input');
        cb.addEventListener('change', () => {
            if (cb.checked) {
                if (!state.selectedCollections.includes(col)) state.selectedCollections.push(col);
            } else {
                state.selectedCollections = state.selectedCollections.filter(c => c !== col);
            }
        });
        container.appendChild(item);
    }
}

// ─── Suggested Questions ───────────────────────────────────────────
async function loadSuggestedQuestions() {
    const container = document.getElementById('suggested-questions');
    container.innerHTML = '';
    for (let i = 0; i < 3; i++) {
        const skel = document.createElement('div');
        skel.className = 'skeleton-line';
        skel.style.width = `${140 + Math.random() * 100}px`;
        skel.style.height = '38px';
        skel.style.borderRadius = '100px';
        container.appendChild(skel);
    }

    try {
        const res = await fetch('/suggested-questions');
        const data = await res.json();
        container.innerHTML = '';
        for (const q of (data.questions || [])) {
            const pill = document.createElement('button');
            pill.className = 'suggested-pill';
            pill.textContent = q;
            pill.addEventListener('click', () => {
                if (!state.isStreaming) sendMessage(q);
            });
            container.appendChild(pill);
        }
    } catch (e) {
        container.innerHTML = '';
        console.error('Failed to load suggested questions:', e);
    }
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
    const toggle = document.getElementById('toggle-sources');
    if (!toggle || !toggle.checked) return;
    if (!sources || sources.length === 0) return;

    const container = document.createElement('div');
    container.className = 'sources-list';

    sources.forEach((src, i) => {
        const ref = document.createElement('div');
        ref.className = 'source-ref';

        const idxSpan = document.createElement('span');
        idxSpan.className = 'source-ref-idx';
        idxSpan.textContent = `[${src.index || (i + 1)}]`;

        const nameSpan = document.createElement('span');
        nameSpan.className = 'source-ref-name';
        nameSpan.textContent = src.filename || 'desconocido';

        ref.appendChild(idxSpan);
        ref.appendChild(nameSpan);

        if (src.date_iso) {
            const dateSpan = document.createElement('span');
            dateSpan.className = 'source-ref-date';
            dateSpan.textContent = src.date_iso;
            ref.appendChild(dateSpan);
        }

        container.appendChild(ref);
    });

    bubble.appendChild(container);
}

function renderFollowupPills(msgDiv, followups) {
    if (!followups || followups.length === 0) return;
    const container = document.createElement('div');
    container.className = 'followup-container';

    for (const q of followups) {
        const pill = document.createElement('button');
        pill.className = 'followup-pill';
        pill.textContent = q;
        pill.addEventListener('click', () => {
            if (!state.isStreaming) {
                sendMessage(q);
            }
        });
        container.appendChild(pill);
    }

    msgDiv.appendChild(container);
}

// ─── Disable old follow-ups ────────────────────────────────────────
function disableOldFollowups() {
    document.querySelectorAll('.followup-pill').forEach(pill => {
        pill.disabled = true;
    });
}

// ─── Send Message (streaming) ──────────────────────────────────────
async function sendMessage(query) {
    if (state.isStreaming || !query.trim()) return;
    state.isStreaming = true;
    sendBtn.disabled = true;
    userInput.value = '';
    userInput.style.height = 'auto';

    disableOldFollowups();
    hideWelcomeScreen();

    state.messages.push({ role: 'user', content: query });
    renderUserMessage(query);

    const history = state.messages
        .slice(-10)
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content }));

    const body = {
        query: query,
        k: estimateK(query),
        collections: state.selectedCollections.length ? state.selectedCollections : null,
        use_llm_expansion: query.trim().split(/\s+/).length >= 5,
        history: history.length > 1 ? history.slice(0, -1) : null,
    };

    const typingEl = showTypingIndicator();

    try {
        const response = await fetch('/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        typingEl.remove();
        const { msgDiv, bubble, content } = createAssistantBubble();

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = '';
        let sources = [];
        let followups = [];
        let buffer = '';
        let done = false;

        // Throttled rendering: parse markdown at most every 80ms during streaming
        let renderTimer = null;
        let needsRender = false;

        function doRender(final) {
            const sourcesToggle = document.getElementById('toggle-sources');
            const renderText = (sourcesToggle && !sourcesToggle.checked)
                ? fullAnswer.replace(/\[\d+\]/g, '')
                : fullAnswer;
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
                }, 80);
            }
        }

        while (!done) {
            const { done: readerDone, value } = await reader.read();
            if (readerDone) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                // SSE: lines starting with "data: " carry payload.
                // Consecutive data lines before a blank line form one
                // logical event — rejoin them with newlines so multi-line
                // LLM chunks arrive intact.
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6);

                if (payload === '[END]') {
                    done = true;
                    break;
                }
                if (payload.startsWith('[ERROR]')) {
                    fullAnswer += payload.slice(7).trim();
                    doRender(true);
                    continue;
                }
                if (payload === '[DONE]') {
                    doRender(true);
                    continue;
                }
                if (payload.startsWith('[SOURCES] ')) {
                    try { sources = JSON.parse(payload.slice(10)); } catch(e) {}
                    renderSourceBadges(bubble, sources);
                    scrollToBottom();
                    continue;
                }
                if (payload.startsWith('[FOLLOWUPS] ')) {
                    try { followups = JSON.parse(payload.slice(12)); } catch(e) {}
                    renderFollowupPills(msgDiv, followups);
                    scrollToBottom();
                    continue;
                }

                // Regular text chunk — throttled render
                fullAnswer += payload;
                scheduleRender();
            }
        }

        // Final render to ensure nothing is missed
        if (needsRender || renderTimer) {
            doRender(true);
        }

        state.messages.push({
            role: 'assistant',
            content: fullAnswer,
            sources,
            followups,
        });

        scrollToBottom();

    } catch (e) {
        typingEl.remove();
        const { msgDiv, bubble, content } = createAssistantBubble();
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

// Theme buttons in settings
document.querySelectorAll('.theme-option').forEach(btn => {
    btn.addEventListener('click', () => {
        applyTheme(btn.dataset.themeVal);
    });
});

// Clear chat
function clearChat() {
    if (state.messages.length > 0 && !confirm('¿Se borrará todo el historial de chat. Continuar?')) {
        return;
    }
    state.messages = [];
    const messages = chatArea.querySelectorAll('.message');
    messages.forEach(m => m.remove());
    showWelcomeScreen();
    loadSuggestedQuestions();
    toggleSettings(false);
}
document.getElementById('btn-clear-chat').addEventListener('click', clearChat);

// Export chat
document.getElementById('btn-export-chat').addEventListener('click', () => {
    if (state.messages.length === 0) return;
    const data = JSON.stringify(state.messages, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat_export_${new Date().toISOString().slice(0,19).replace(/[:-]/g,'')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toggleSettings(false);
});

// ─── Mobile Keyboard Handling ──────────────────────────────────────
if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', () => {
        setTimeout(() => scrollToBottom(), 100);
    });
}

// ─── Sources Toggle Persistence ─────────────────────────────────────
function initSourcesToggle() {
    const toggle = document.getElementById('toggle-sources');
    const stored = localStorage.getItem('showSources');
    if (stored !== null) toggle.checked = stored !== 'false';
    toggle.addEventListener('change', () => {
        localStorage.setItem('showSources', toggle.checked);
    });
}

// ─── Init ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    applyTheme(getStoredTheme());
    loadCollections();
    loadSuggestedQuestions();
    initSourcesToggle();
    userInput.focus();
});
