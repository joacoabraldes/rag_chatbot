// Entry point. Wires the modules to the DOM once it's ready.

import { clearMessages, hideWelcomeScreen, initInput, sendMessage } from './modules/chat.js';
import { initDrawer } from './modules/drawer.js';
import { initSettings } from './modules/settings.js';
import { renderSuggestions } from './modules/suggestions.js';
import { initTheme } from './modules/theme.js';

function bootstrap() {
    initTheme();
    initDrawer();
    initSettings({ onClear: clearMessages });
    initInput();
    renderSuggestions({
        onSelect: (q) => {
            hideWelcomeScreen();
            sendMessage(q);
        },
    });
    document.getElementById('user-input').focus();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
} else {
    bootstrap();
}
