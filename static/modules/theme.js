// Theme management — system / light / dark with localStorage persistence.

const THEME_KEY = 'rag-chatbot-theme';

export function getStoredTheme() {
    return localStorage.getItem(THEME_KEY) || 'system';
}

export function applyTheme(mode) {
    document.documentElement.dataset.theme = mode;
    localStorage.setItem(THEME_KEY, mode);
    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.themeVal === mode);
    });
}

export function initTheme() {
    applyTheme(getStoredTheme());

    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.addEventListener('click', () => applyTheme(btn.dataset.themeVal));
    });

    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
            // Bumping dataset triggers a re-evaluation of the CSS media query block.
            if (getStoredTheme() === 'system') {
                document.documentElement.dataset.theme = 'system';
            }
        });
    }
}
