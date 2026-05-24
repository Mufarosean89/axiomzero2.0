/* ═══════════════════════════════════════════════
   Axiom Zero — Web App
   Lean syntax highlighting, sample selector,
   error display, line numbers, keyboard shortcuts
   ═══════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── DOM References ──────────────────────────────────

    const pythonInput = document.getElementById('python-input');
    const compileBtn = document.getElementById('compile-btn');
    const leanCode = document.getElementById('lean-code');
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorPanel = document.getElementById('error-panel');
    const errorDetails = document.getElementById('error-details');
    const errorContent = document.getElementById('error-content');
    const errorToggle = document.getElementById('error-toggle');
    const outputStats = document.getElementById('output-stats');
    const clearBtn = document.getElementById('clear-btn');
    const samplesBtn = document.getElementById('samples-btn');
    const samplesModal = document.getElementById('samples-modal');
    const samplesModalClose = document.getElementById('samples-modal-close');
    const sampleList = document.getElementById('sample-list');
    const inputLineNumbers = document.getElementById('input-line-numbers');
    const outputLineNumbers = document.getElementById('output-line-numbers');
    const outputWrapper = document.querySelector('.output-wrapper');

    // ── Samples Data ────────────────────────────────────

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

    // ── Lean 4 Syntax Highlighting ──────────────────────

    // Register a custom Lean 4 language for highlight.js
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
                        // Multi-line comments
                        {
                            className: 'comment',
                            begin: /\/-/,
                            end: /-\//,
                            contains: []
                        },
                        // Single-line comments
                        {
                            className: 'comment',
                            begin: /--/,
                            end: /$/,
                            contains: []
                        },
                        // Strings
                        {
                            className: 'string',
                            variants: [
                                { begin: /"/, end: /"/, contains: [{ begin: /\\\\/ }] },
                                { begin: /'/, end: /'/ }
                            ]
                        },
                        // Numbers
                        {
                            className: 'number',
                            variants: [
                                { begin: /\b\d+\.\d+/ },
                                { begin: /\b\d+/ }
                            ]
                        },
                        // Type variables (capitalized identifiers)
                        {
                            className: 'type',
                            begin: /\b[A-Z][a-zA-Z0-9_']*/
                        },
                        // Theorem names after "theorem"
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

    // Fallback: manual Lean syntax highlighting when hljs is unavailable
    function highlightLeanFallback(code) {
        if (!code) return '';

        let h = code
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Comments: /- ... -/
        h = h.replace(/(\/-[\s\S]*?-\/)/g, '<span class="hljs-comment">$1</span>');

        // Comments: -- to end of line
        h = h.replace(/(--[^\n]*)/g, '<span class="hljs-comment">$1</span>');

        // Keywords
        const keywords = [
            'theorem', 'lemma', 'example', 'def', 'import', 'open', 'by',
            'have', 'show', 'let', 'in', 'match', 'with', 'fun', 'calc',
            'sorry', 'admit', 'exact', 'apply', 'refine', 'simp', 'rw',
            'rfl', 'trivial', 'assumption', 'omega', 'decide',
            'induction', 'cases', 'constructor', 'injection',
            'left', 'right', 'split', 'if', 'then', 'else',
            'forall', 'exists', 'where', 'from', 'end'
        ];
        const kwRegex = new RegExp('\\b(' + keywords.join('|') + ')\\b', 'g');
        h = h.replace(kwRegex, '<span class="hljs-keyword">$1</span>');

        // Theorem names after keyword
        h = h.replace(/(theorem|lemma|example|def)\s+([a-zA-Z_][a-zA-Z0-9_']*)/g,
            '$1 <span class="hljs-title">$2</span>');

        // Types (Nat, ℕ, Int, ℤ, etc.)
        const types = ['Nat', 'ℕ', 'Int', 'ℤ', 'Bool', 'String', 'List', 'Option',
                       'Type', 'Prop', 'Real', 'ℝ', 'True', 'False'];
        const typeRegex = new RegExp('\\b(' + types.join('|') + ')\\b', 'g');
        h = h.replace(typeRegex, '<span class="hljs-type">$1</span>');

        // Numbers
        h = h.replace(/\b(\d+\.?\d*)\b/g, '<span class="hljs-number">$1</span>');

        // Strings
        h = h.replace(/("(?:[^"\\]|\\.)*")/g, '<span class="hljs-string">$1</span>');

        return h;
    }

    function highlightLean(code) {
        if (typeof hljs !== 'undefined') {
            try {
                const highlighted = hljs.highlight(code, { language: 'lean' });
                return highlighted.value;
            } catch (e) {
                // fall through to fallback
            }
        }
        return highlightLeanFallback(code);
    }

    // Register Lean language on load
    registerLeanLanguage();

    // ── Line Number Sync ────────────────────────────────

    function updateLineNumbers(textarea, lineNumbersEl) {
        const lines = textarea.value.split('\n');
        const count = Math.max(lines.length, 1);
        let html = '';
        for (let i = 1; i <= count; i++) {
            html += '<div class="line-number" data-ln="' + i + '">' + i + '</div>';
        }
        lineNumbersEl.innerHTML = html;
    }

    // Sync input line numbers on input and on load
    pythonInput.addEventListener('input', function () {
        updateLineNumbers(pythonInput, inputLineNumbers);
    });

    function syncOutputLineNumbers(code) {
        if (!code) {
            outputLineNumbers.innerHTML = '<div class="line-number" data-ln="1">1</div>';
            return;
        }
        const lines = code.split('\n');
        let html = '';
        for (let i = 1; i <= lines.length; i++) {
            html += '<div class="line-number" data-ln="' + i + '">' + i + '</div>';
        }
        outputLineNumbers.innerHTML = html;
    }

    // Sync textarea scroll with line numbers
    pythonInput.addEventListener('scroll', function () {
        inputLineNumbers.scrollTop = pythonInput.scrollTop;
    });

    // Tab support in textarea
    pythonInput.addEventListener('keydown', function (e) {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = this.selectionStart;
            const end = this.selectionEnd;
            this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
            this.selectionStart = this.selectionEnd = start + 4;
            updateLineNumbers(pythonInput, inputLineNumbers);
        }
    });

    // ── Sample Selector ─────────────────────────────────

    function populateSamples() {
        sampleList.innerHTML = '';
        SAMPLES.forEach(function (sample, index) {
            const item = document.createElement('button');
            item.className = 'sample-item';
            item.innerHTML =
                '<div class="sample-item-icon"><i class="' + sample.icon + '"></i></div>' +
                '<div class="sample-item-content">' +
                    '<div class="sample-item-title">' + sample.title + '</div>' +
                    '<div class="sample-item-desc">' + sample.desc + '</div>' +
                '</div>' +
                '<span class="sample-item-tag ' + sample.tagClass + '">' + sample.tag + '</span>';
            item.addEventListener('click', function () {
                pythonInput.value = sample.source;
                updateLineNumbers(pythonInput, inputLineNumbers);
                closeModal();
                // Focus input
                pythonInput.focus();
            });
            sampleList.appendChild(item);
        });
    }

    function openModal() {
        samplesModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        samplesModal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    samplesBtn.addEventListener('click', openModal);
    samplesModalClose.addEventListener('click', closeModal);
    samplesModal.querySelector('.modal-backdrop').addEventListener('click', closeModal);

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && !samplesModal.classList.contains('hidden')) {
            closeModal();
        }
    });

    populateSamples();

    // ── Clear Button ────────────────────────────────────

    clearBtn.addEventListener('click', function () {
        pythonInput.value = '';
        pythonInput.focus();
        updateLineNumbers(pythonInput, inputLineNumbers);
    });

    // ── Keyboard Shortcut ───────────────────────────────

    pythonInput.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            compile();
        }
    });

    // ── Compile ─────────────────────────────────────────

    function compile() {
        let source = pythonInput.value.trim();
        if (!source) {
            // Load default sample on empty
            pythonInput.value = SAMPLES[0].source;
            updateLineNumbers(pythonInput, inputLineNumbers);
            source = pythonInput.value.trim();
        }

        // UI: compiling state
        compileBtn.disabled = true;
        copyBtn.disabled = true;
        downloadBtn.disabled = true;
        loadingIndicator.classList.remove('hidden');
        outputWrapper.classList.add('compiling');
        errorPanel.classList.add('hidden');
        errorDetails.classList.add('hidden');
        errorToggle.classList.remove('expanded');
        outputStats.textContent = '';
        outputStats.className = 'output-stats';

        fetch('/api/compile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: source })
        })
        .then(function (response) {
            return response.json().then(function (data) {
                return { ok: response.ok, status: response.status, data: data };
            });
        })
        .then(function (result) {
            if (!result.ok) {
                var err = new Error(result.data.error || 'Unknown compilation error (status ' + result.status + ')');
                err._traceback = result.data.traceback || null;
                throw err;
            }

            const code = result.data.lean_code || '';
            const highlighted = highlightLean(code);
            leanCode.innerHTML = highlighted;
            syncOutputLineNumbers(code);

            // Stats
            const theoremCount = (code.match(/theorem\s+/g) || []).length;
            const lineCount = code.split('\n').length;
            outputStats.textContent = theoremCount + ' theorems · ' + lineCount + ' lines';
            outputStats.className = 'output-stats success';

            copyBtn.disabled = false;
            downloadBtn.disabled = false;

            // Hide error if previously shown
            errorPanel.classList.add('hidden');
        })
        .catch(function (error) {
            // Show error panel
            var errorMsg = error.message || 'An unexpected error occurred.';

            // Try to get structured error data from the fetch response
            // (the error was re-thrown in .then(), so we access result.data from the closure)
            leanCode.innerHTML = '<span style="color: var(--text-muted); font-style: italic;">Compilation failed &mdash; see error below.</span>';
            syncOutputLineNumbers('');
            outputStats.textContent = 'Error';
            outputStats.className = 'output-stats';

            // Build full error content with optional traceback
            var fullError = errorMsg;
            if (error._traceback) {
                fullError += '\n\nTraceback:\n' + error._traceback;
            }

            errorContent.textContent = fullError;
            errorPanel.classList.remove('hidden');
            errorDetails.classList.add('hidden');
            errorToggle.classList.remove('expanded');

            copyBtn.disabled = true;
            downloadBtn.disabled = true;
        })
        .finally(function () {
            compileBtn.disabled = false;
            loadingIndicator.classList.add('hidden');
            outputWrapper.classList.remove('compiling');
        });
    }

    compileBtn.addEventListener('click', compile);

    // ── Error Toggle ────────────────────────────────────

    errorToggle.addEventListener('click', function () {
        const isHidden = errorDetails.classList.contains('hidden');
        if (isHidden) {
            errorDetails.classList.remove('hidden');
            errorToggle.classList.add('expanded');
        } else {
            errorDetails.classList.add('hidden');
            errorToggle.classList.remove('expanded');
        }
    });

    // ── Copy ────────────────────────────────────────────

    copyBtn.addEventListener('click', function () {
        const text = leanCode.textContent || leanCode.innerText || '';
        if (!text) return;

        navigator.clipboard.writeText(text).then(function () {
            showToast('Proof copied to clipboard', 'success');
        }).catch(function () {
            // Fallback
            const range = document.createRange();
            range.selectNode(leanCode);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();
            showToast('Proof copied to clipboard', 'success');
        });
    });

    // ── Download ────────────────────────────────────────

    downloadBtn.addEventListener('click', function () {
        const text = leanCode.textContent || leanCode.innerText || '';
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

    // ── Toast ───────────────────────────────────────────

    function showToast(message, type) {
        const existing = document.querySelector('.toast');
        if (existing) existing.remove();

        const icon = type === 'success' ? 'fa-solid fa-check-circle' : 'fa-solid fa-circle-exclamation';
        const toast = document.createElement('div');
        toast.className = 'toast ' + (type || '');
        toast.innerHTML = '<i class="' + icon + '"></i> ' + message;
        document.body.appendChild(toast);

        setTimeout(function () {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(function () { toast.remove(); }, 300);
        }, 2500);
    }

    // ── Init ────────────────────────────────────────────

    // Set default sample if empty
    if (!pythonInput.value.trim()) {
        pythonInput.value = SAMPLES[0].source;
    }
    updateLineNumbers(pythonInput, inputLineNumbers);

    console.log('Axiom Zero Web App initialized');
})();
