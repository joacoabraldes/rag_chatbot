// Empty-state suggestions — clickable example questions shown when the
// chat is empty. Bites off the cold-start problem for new users.

const EXAMPLES = [
    {
        label: 'Dólar hoy',
        query: '¿Cuáles son las cotizaciones del dólar de hoy?',
    },
    {
        label: 'Último informe',
        query: '¿Cuál es el informe más reciente y qué temas cubre?',
    },
    {
        label: 'Brecha cambiaria',
        query: '¿Cómo evolucionó la brecha cambiaria en las últimas dos semanas?',
    },
    {
        label: 'MEP vs CCL',
        query: 'Compará el MEP y el CCL del último mes.',
    },
];

export function renderSuggestions({ onSelect }) {
    const container = document.getElementById('suggestions');
    if (!container) return;
    container.innerHTML = '';
    EXAMPLES.forEach((ex, i) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'suggestion-chip';
        btn.style.animationDelay = `${0.05 * (i + 1)}s`;
        btn.innerHTML = `
            <span class="suggestion-chip-label">${ex.label}</span>
            <span class="suggestion-chip-query">${ex.query}</span>
        `;
        btn.addEventListener('click', () => onSelect(ex.query));
        container.appendChild(btn);
    });
}
