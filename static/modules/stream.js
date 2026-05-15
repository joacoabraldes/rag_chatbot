// Parse the SSE stream from /chat. Yields typed events:
//   { type: 'token', text }
//   { type: 'meta',  data }
//   { type: 'sources', data }
//   { type: 'followups', data }
//   { type: 'error', text }
//   { type: 'done' }
//   { type: 'end' }
//
// Keeps the SSE wire format the backend already uses; centralizes the
// payload-dispatch logic so chat.js can `for await` it.

async function* readLines(reader) {
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            if (buffer) yield buffer;
            return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) yield line;
    }
}

export async function* parseSSE(response) {
    const reader = response.body.getReader();
    for await (const line of readLines(reader)) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);

        if (payload === '[END]') {
            yield { type: 'end' };
            return;
        }
        if (payload === '[DONE]') {
            yield { type: 'done' };
            continue;
        }
        if (payload.startsWith('[ERROR]')) {
            yield { type: 'error', text: payload.slice(7).trim() };
            continue;
        }
        if (payload.startsWith('[META] ')) {
            try {
                yield { type: 'meta', data: JSON.parse(payload.slice(7)) };
            } catch (_) { /* ignore malformed meta */ }
            continue;
        }
        if (payload.startsWith('[SOURCES] ')) {
            try {
                yield { type: 'sources', data: JSON.parse(payload.slice(10)) };
            } catch (_) { /* ignore */ }
            continue;
        }
        if (payload.startsWith('[FOLLOWUPS] ')) {
            try {
                yield { type: 'followups', data: JSON.parse(payload.slice(12)) };
            } catch (_) { /* ignore */ }
            continue;
        }

        // Plain content token. Backend JSON-encodes each token to escape
        // newlines / quotes; fall back to raw payload if parsing fails.
        let token = payload;
        try { token = JSON.parse(payload); } catch (_) {}
        yield { type: 'token', text: token };
    }
}
