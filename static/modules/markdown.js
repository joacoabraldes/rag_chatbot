// Incremental markdown renderer.
//
// Why this exists: the old code did `marked.parse(fullAnswer)` on every
// streaming token. For a 2k-char response that's hundreds of full parses
// plus a full innerHTML replace each time — janks badly on mobile.
//
// Strategy: split the response on the last paragraph boundary (\n\n).
// Everything before is "committed" — parsed once, cached as HTML, never
// reparsed unless a new paragraph closes. Only the trailing partial
// paragraph is reparsed on each token. innerHTML write is still cheap
// when most of the document hasn't changed.

const PARAGRAPH_BREAK = /\n\n/;

export class IncrementalMarkdown {
    constructor(target) {
        this.target = target;
        this.committedText = '';
        this.committedHTML = '';
    }

    update(fullText) {
        const lastBreak = fullText.lastIndexOf('\n\n');

        if (lastBreak === -1) {
            // No paragraph break yet — single tail to parse.
            this.target.innerHTML = window.marked.parse(fullText);
            return;
        }

        const stable = fullText.slice(0, lastBreak);
        const tail = fullText.slice(lastBreak); // includes the leading \n\n

        if (stable !== this.committedText) {
            this.committedHTML = window.marked.parse(stable);
            this.committedText = stable;
        }

        // Parse the tail in isolation; concatenate with the cached HTML.
        // The leading \n\n keeps tail-parsing standalone (no list/blockquote
        // continuation from the committed section bleeds across).
        this.target.innerHTML = this.committedHTML + window.marked.parse(tail);
    }

    finalize(fullText) {
        // One last full parse so any partial syntax (open code fence, etc.)
        // gets resolved correctly in the final state.
        this.target.innerHTML = window.marked.parse(fullText);
        this.committedText = fullText;
        this.committedHTML = this.target.innerHTML;
    }
}
