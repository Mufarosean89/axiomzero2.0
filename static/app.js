(function () {
    'use strict';

    const dom = {
        pythonInput: document.getElementById('python-input'),
        compileBtn: document.getElementById('compile-btn'),
        leanCode: document.getElementById('lean-code'),
        copyBtn: document.getElementById('copy-btn'),
        downloadBtn: document.getElementById('download-btn'),
        loadingIndicator: document.getElementById('loading-indicator'),
        errorPanel: document.getElementById('error-panel'),
        errorDetails: document.getElementById('error-details'),
        errorContent: document.getElementById('error-content'),
        errorToggle: document.getElementById('error-toggle'),
        outputStats: document.getElementById('output-stats'),
        clearBtn: document.getElementById('clear-btn'),
        samplesBtn: document.getElementById('samples-btn'),
        samplesModal: document.getElementById('samples-modal'),
        samplesModalClose: document.getElementById('samples-modal-close'),
        sampleList: document.getElementById('sample-list'),
        inputLineNumbers: document.getElementById('input-line-numbers'),
        outputLineNumbers: document.getElementById('output-line-numbers'),
        outputWrapper: document.querySelector('.output-wrapper'),
    };

    const SAMPLES = [
        {
            title: 'Absolute Value',
            desc: 'Classic absolute value with pre/post-conditions',
            tag: 'Control Flow',
            tagClass: 'control',
            icon: 'fa-solid fa-code',
            source: `@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x`
        },
        {
            title: 'Addition',
            desc: 'Simple addition with bound constraints',
            tag: 'Arithmetic',
            tagClass: 'arithmetic',
            icon: 'fa-solid fa-plus',
            source: `@requires("x >= 0")
@requires("y >= 0")
@ensures("result >= 0")
def add_bounded(x: int, y: int) -> int:
    return x + y`
        },
        {
            title: 'Factorial',
            desc: 'Recursive factorial with pre/post-conditions',
            tag: 'Control Flow',
            tagClass: 'control',
            icon: 'fa-solid fa-arrow-trend-up',
            source: `@requires("n >= 0")
@ensures("result >= 0")
def factorial(n: int) -> int:
    if n == 0:
        return 1
    return n * factorial(n - 1)`
        },
        {
            title: 'List Head',
            desc: 'Safely get the first element of a list',
            tag: 'Control Flow',
            tagClass: 'control',
            icon: 'fa-solid fa-list',
            source: `@requires("len(xs) > 0")
@ensures("result in xs")
def head(xs: list) -> int:
    return xs[0]`
        },
        {
            title: 'ReLU Forward',
            desc: 'Neural network activation function',
            tag: 'Tensor',
            tagClass: 'tensor',
            icon: 'fa-solid fa-brain',
            source: `import torch
import torch.nn.functional as F

@requires("x > 0")
@ensures("result > 0")
def relu_forward(x: torch.Tensor) -> torch.Tensor:
    result = F.relu(x)
    return result`
        },
        {
            title: 'Simple MLP',
            desc: 'Two-layer MLP forward pass with shape specs',
            tag: 'Tensor',
            tagClass: 'tensor',
            icon: 'fa-solid fa-network-wired',
            source: `import torch
import torch.nn.functional as F

class SimpleMLP:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        self.w1 = torch.randn(in_dim, hidden_dim)
        self.w2 = torch.randn(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.w1
        h = F.relu(h)
        out = h @ self.w2
        return out`
        }
    ];

    function registerLeanLanguage() {
        if (typeof hljs === 'undefined') return;
        try {
            hljs.registerLanguage('lean', function () {
                return {
                    name: 'Lean 4',
                    aliases: ['lean4', 'lean'],
                    keywords: {
                        $pattern: /[a-zA-Z_]\w*/,
                        keyword: [
                            'theorem', 'lemma', 'example', 'def', 'inductive',
                            'structure', 'class', 'instance', 'axiom', 'opaque',
                            'let', 'in', 'have', 'show', 'by', 'calc',
                            'match', 'with', 'fun', 'forall', 'exists',
                            'if', 'then', 'else', 'do', 'return',
                            'import', 'open', 'namespace', 'section', 'end',
                            'variable', 'variables', 'parameter', 'parameters',
                            'mutual', 'where', 'from', 'using', 'hiding',
                            'set_option', 'run_cmd', 'macro', 'elab',
                            'private', 'protected', 'public', 'partial',
                            'noncomputable', 'unsafe', 'inline',
                            'sorry', 'admit', 'exact', 'apply', 'refine',
                            'simp', 'rw', 'rfl', 'trivial', 'assumption',
                            'omega', 'decide', 'arith', 'native_decide',
                            'induction', 'cases', 'constructor', 'injection',
                            'left', 'right', 'split', 'solve_by_elim',
                            'done', 'qed', 'skip', 'all_goals',
                            'try', 'repeat', 'first', 'focus',
                            'observes'
                        ],
                        type: [
                            'Nat', 'ℕ', 'Int', 'ℤ', 'Rat', 'ℚ', 'Real', 'ℝ',
                            'Complex', 'ℂ', 'Bool', 'True', 'False',
                            'Unit', 'String', 'Char', 'List', 'Array',
                            'Option', 'Sum', 'Prod', 'And', 'Or', 'Not',
                            'Type', 'Prop', 'Sort', 'Set',
                            'Fin', 'Vec', 'Matrix', 'Tensor',
                            'Function', 'Pi', 'Sigma'
                        ],
                        literal: [
                            'true', 'false', 'none', 'some',
                            '0', '1', 'nil', 'cons'
                        ],
                        built_in: [
                            'Nat', 'ℕ', 'Int', 'ℤ', 'Bool', 'String',
                            'simp', 'omega', 'rfl', 'trivial'
                        ]
                    },
                    contains: [
                        { className: 'comment', begin: /\/-/, end: /-\//, contains: [] },
                        { className: 'comment', begin: /--/, end: /$/, contains: [] },
                        {
                            className: 'string',
                            variants: [
                                { begin: /"/, end: /"/, contains: [{ begin: /\\\\/ }] },
                                { begin: /'/, end: /'/ }
                            ]
                        },
                        {
                            className: 'number',
                            variants: [
                                { begin: /\b\d+\.\d+/ },
                                { begin: /\b\d+/ }
                            ]
                        },
                        { className: 'type', begin: /\b[A-Z][a-zA-Z0-9_']*/ },
                        {
                            className: 'title',
                            begin: /(?:theorem|lemma|example|def)\s+/,
                            end: /\s+|:/,
                            excludeBegin: true,
                            excludeEnd: true,
                            relevance: 5
                        }
                    ]
                };
            });
        } catch (e) {
            console.warn('Failed to register Lean language:', e);
        }
    }

    function highlightLeanFallback(code) {
        if (!code) return '';

        const escapeHtml = (s) => s.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        let h = escapeHtml(code);

        h = h.replace(/(\/-[\s\S]*?-\/)/g, '<span class="hljs-comment">$1</span>');
        h = h.replace(/(--[^\n]*)/g, '<span class="hljs-comment">$1</span>');

        const keywords = [
            'theorem', 'lemma', 'example', 'def', 'import', 'open', 'by',
            'have', 'show', 'let', 'in', 'match', 'with', 'fun', 'calc',
            'sorry', 'admit', 'exact', 'apply', 'refine', 'simp', 'rw',
            'rfl', 'trivial', 'assumption', 'omega', 'decide',
            'induction', 'cases', 'constructor', 'injection',
            'left', 'right', 'split', 'if', 'then', 'else',
            'forall', 'exists', 'where', 'from', 'end'
        ];
        h = h.replace(new RegExp('\\b(' + keywords.join('|') + ')\\b', 'g'),
            '<span class="hljs-keyword">$1</span>');

        h = h.replace(/(theorem|lemma|example|def)\s+([a-zA-Z_][a-zA-Z0-9_']*)/g,
            '$1 <span class="hljs-title">$2</span>');

        const types = ['Nat', 'ℕ', 'Int', 'ℤ', 'Bool', 'String', 'List', 'Option',
            'Type', 'Prop', 'Real', 'ℝ', 'True', 'False'];
        h = h.replace(new RegExp('\\b(' + types.join('|') + ')\\b', 'g'),
            '<span class="hljs-type">$1</span>');

        h = h.replace(/\b(\d+\.?\d*)\b/g, '<span class="hljs-number">$1</span>');
        h = h.replace(/("(?:[^"\\]|\\.)*")/g, '<span class="hljs-string">$1</span>');

        return h;
    }

    function highlightLean(code) {
        if (typeof hljs !== 'undefined') {
            try {
                return hljs.highlight(code, { language: 'lean' }).value;
            } catch (e) {
            }
        }
        return highlightLeanFallback(code);
    }

    registerLeanLanguage();

    function updateLineNumbers(textarea, lineNumbersEl) {
        const count = Math.max(textarea.value.split('\n').length, 1);
        lineNumbersEl.innerHTML = Array.from({ length: count }, (_, i) =>
            `<div class="line-number" data-ln="${i + 1}">${i + 1}</div>`
        ).join('');
    }

    dom.pythonInput.addEventListener('input', () =>
        updateLineNumbers(dom.pythonInput, dom.inputLineNumbers)
    );

    function syncOutputLineNumbers(code) {
        if (!code) {
            dom.outputLineNumbers.innerHTML = '<div class="line-number" data-ln="1">1</div>';
            return;
        }
        const lines = code.split('\n');
        dom.outputLineNumbers.innerHTML = Array.from({ length: lines.length }, (_, i) =>
            `<div class="line-number" data-ln="${i + 1}">${i + 1}</div>`
        ).join('');
    }

    dom.pythonInput.addEventListener('scroll', () => {
        dom.inputLineNumbers.scrollTop = dom.pythonInput.scrollTop;
    });

    dom.pythonInput.addEventListener('keydown', function (e) {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = this.selectionStart;
            const end = this.selectionEnd;
            this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
            this.selectionStart = this.selectionEnd = start + 4;
            updateLineNumbers(dom.pythonInput, dom.inputLineNumbers);
        }
    });

    function populateSamples() {
        dom.sampleList.innerHTML = '';
        SAMPLES.forEach((sample) => {
            const item = document.createElement('button');
            item.className = 'sample-item';
            item.innerHTML = `
                <div class="sample-item-icon"><i class="${sample.icon}"></i></div>
                <div class="sample-item-content">
                    <div class="sample-item-title">${sample.title}</div>
                    <div class="sample-item-desc">${sample.desc}</div>
                </div>
                <span class="sample-item-tag ${sample.tagClass}">${sample.tag}</span>`;
            item.addEventListener('click', () => {
                dom.pythonInput.value = sample.source;
                updateLineNumbers(dom.pythonInput, dom.inputLineNumbers);
                closeModal();
                dom.pythonInput.focus();
            });
            dom.sampleList.appendChild(item);
        });
    }

    function openModal() {
        dom.samplesModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        dom.samplesModal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    dom.samplesBtn.addEventListener('click', openModal);
    dom.samplesModalClose.addEventListener('click', closeModal);
    dom.samplesModal.querySelector('.modal-backdrop').addEventListener('click', closeModal);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !dom.samplesModal.classList.contains('hidden')) {
            closeModal();
        }
    });

    populateSamples();

    dom.clearBtn.addEventListener('click', () => {
        dom.pythonInput.value = '';
        dom.pythonInput.focus();
        updateLineNumbers(dom.pythonInput, dom.inputLineNumbers);
    });

    dom.pythonInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            compile();
        }
    });

    function compile() {
        let source = dom.pythonInput.value.trim();
        if (!source) {
            dom.pythonInput.value = SAMPLES[0].source;
            updateLineNumbers(dom.pythonInput, dom.inputLineNumbers);
            source = dom.pythonInput.value.trim();
        }

        dom.compileBtn.disabled = true;
        dom.copyBtn.disabled = true;
        dom.downloadBtn.disabled = true;
        dom.loadingIndicator.classList.remove('hidden');
        dom.outputWrapper.classList.add('compiling');
        dom.errorPanel.classList.add('hidden');
        dom.errorDetails.classList.add('hidden');
        dom.errorToggle.classList.remove('expanded');
        dom.outputStats.textContent = '';
        dom.outputStats.className = 'output-stats';

        fetch('/api/compile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source })
        })
            .then((response) =>
                response.json().then((data) => ({ ok: response.ok, status: response.status, data }))
            )
            .then((result) => {
                if (!result.ok) {
                    const err = new Error(result.data.error || `Unknown compilation error (status ${result.status})`);
                    err.traceback = result.data.traceback || null;
                    throw err;
                }

                const code = result.data.lean_code || '';
                dom.leanCode.innerHTML = highlightLean(code);
                syncOutputLineNumbers(code);

                const theoremCount = (code.match(/theorem\s+/g) || []).length;
                const lineCount = code.split('\n').length;
                dom.outputStats.textContent = `${theoremCount} theorems \u00b7 ${lineCount} lines`;
                dom.outputStats.className = 'output-stats success';

                dom.copyBtn.disabled = false;
                dom.downloadBtn.disabled = false;
                dom.errorPanel.classList.add('hidden');
            })
            .catch((error) => {
                const errorMsg = error.message || 'An unexpected error occurred.';

                dom.leanCode.innerHTML = '<span style="color: var(--text-muted); font-style: italic;">Compilation failed \u2014 see error below.</span>';
                syncOutputLineNumbers('');
                dom.outputStats.textContent = 'Error';
                dom.outputStats.className = 'output-stats';

                const fullError = error.traceback
                    ? `${errorMsg}\n\nTraceback:\n${error.traceback}`
                    : errorMsg;

                dom.errorContent.textContent = fullError;
                dom.errorPanel.classList.remove('hidden');
                dom.errorDetails.classList.add('hidden');
                dom.errorToggle.classList.remove('expanded');

                dom.copyBtn.disabled = true;
                dom.downloadBtn.disabled = true;
            })
            .finally(() => {
                dom.compileBtn.disabled = false;
                dom.loadingIndicator.classList.add('hidden');
                dom.outputWrapper.classList.remove('compiling');
            });
    }

    dom.compileBtn.addEventListener('click', compile);

    dom.errorToggle.addEventListener('click', () => {
        const isHidden = dom.errorDetails.classList.contains('hidden');
        dom.errorDetails.classList.toggle('hidden');
        dom.errorToggle.classList.toggle('expanded');
    });

    dom.copyBtn.addEventListener('click', () => {
        const text = dom.leanCode.textContent || dom.leanCode.innerText || '';
        if (!text) return;

        navigator.clipboard.writeText(text)
            .then(() => showToast('Proof copied to clipboard', 'success'))
            .catch(() => {
                const range = document.createRange();
                range.selectNode(dom.leanCode);
                const selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
                document.execCommand('copy');
                selection.removeAllRanges();
                showToast('Proof copied to clipboard', 'success');
            });
    });

    dom.downloadBtn.addEventListener('click', () => {
        const text = dom.leanCode.textContent || dom.leanCode.innerText || '';
        if (!text) return;

        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'axiom_zero_proof.lean';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('Proof downloaded', 'success');
    });

    function showToast(message, type = 'success') {
        const existing = document.querySelector('.toast');
        if (existing) existing.remove();

        const icon = type === 'success' ? 'fa-solid fa-check-circle' : 'fa-solid fa-circle-exclamation';
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<i class="${icon}"></i> ${message}`;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 2500);
    }

    if (!dom.pythonInput.value.trim()) {
        dom.pythonInput.value = SAMPLES[0].source;
    }
    updateLineNumbers(dom.pythonInput, dom.inputLineNumbers);
})();
