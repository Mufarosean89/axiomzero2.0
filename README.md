# Axiom Zero

**AlphaZero-Style Proof Automation for Neural Network Verification**

Axiom Zero is a compiler that translates Python/PyTorch source code into formal proofs verified by Lean 4. It uses an AlphaZero-style reinforcement learning agent to automatically discover proofs — no human labelling required. The Lean 4 kernel acts as the ground-truth oracle: a proof either compiles or it doesn't.

---

## Architecture

The system has six horizontal layers that data flows through top to bottom, with one feedback arrow (the training signal) going back up.

```
    Python Source      PyTorch Model      Formal Spec
         |                  |                  |
         v                  v                  v
    ┌─────────┐     ┌──────────────┐     ┌──────────┐
    │   AST   │     │  Abstract    │     │  Proof   │
    │Extractor│◄────│ Interpreter  │     │  State   │
    └────┬────┘     └──────┬───────┘     └────┬─────┘
         │                 │                  │
         └─────────────────┴──────────────────┘
                          │
                          v
              ┌─────────────────────┐
              │   RL Proof Agent    │
              │  AlphaZero-Style    │
              │  (Policy + Value +  │
              │    MCTS Search)     │
              └──────────┬──────────┘
                         │
              ┌──────────┼──────────┐
              v          v          v
        ┌─────────┐ ┌────────┐ ┌──────────┐
        │ Lemma   │ │ Lean 4 │ │ Tactic   │
        │Database │ │ Kernel │ │Compiler  │
        └────┬────┘ └────┬───┘ └────┬─────┘
             │           │          │
             └───────────┴──────────┘
                         │
                         v
              ┌─────────────────────┐
              │  Reward: W = +1     │
              │  (Proof compiles)   │
              └─────────────────────┘
                         │
              (training signal back to agent)
```

### Layer 1 — Input

- **Python source** — ordinary Python functions, classes, and loops
- **PyTorch model** — neural network definitions with tensor operations
- **Formal spec** — mathematical specifications as pre/post-conditions (`@requires`, `@ensures`)

### Layer 2 — Parse

- **AST Extractor** — parses Python source into a normalized intermediate representation (IR)
- **Abstract Interpreter** — infers types and tensor shapes at every program point
- **Proof State Builder** — assembles goals and context from specs + interpreter findings

### Layer 3 — Agent

- **RL Proof Agent** — AlphaZero-style: policy network (tactic selection) + value network (progress estimation) + MCTS (tree search)
- **Dataset Loaders** — supports miniF2F, ProofNet, LeanDojo, and Lean Workbook for cold-start pre-training
- **Training Orchestrator** — supervised behavioural cloning pre-training followed by AlphaZero self-play

### Layer 5 — Verify (with Lemma Database integration)

- **Lemma Database** — searchable 128-dim embedding index of ~60 seed lemmas (arithmetic, order, boolean, list, tensor) with n-gram hash retrieval and hit/miss feedback tracking
- **Lean 4 kernel** — primary proof backend; either it compiles or it doesn't

- **Lean 4 kernel** — primary proof backend; either it compiles or it doesn't
- **Tactic Compiler** — translates agent actions into concrete proof terms
- **Web Interface** — browser-based playground for compiling Python source to Lean 4

### Layer 5 — Reward

- **Win condition W** — +1 if the proof compiles, making Lean the oracle
- **Self-play loop** — continuously generates proofs and trains the networks without human labels

---

## Current Status: Phases 1-4 Complete, Phase 5 Underway

| Module | Status | Tests |
|--------|--------|-------|
| `ast_extractor/` | ✅ Done | 16 |
| `abstract_interpreter/` | ✅ Done | (included above) |
| `spec_ingestion/` | ✅ Done | (included above) |
| `proof_engine/` | 🟡 46/48 | 48 |
| `rl_agent/` | ✅ Done | 42 |
| `lean_compiler/` | ✅ Done | 16 |
| **Total** | **~122 tests (120 passing)** | **16 + 48 + 42 + 16** |

### What's Built

#### Phase 1 — Parser Pipeline
- Python source → normalized IR (functions, classes, loops, conditionals, tensor ops)
- Type inference and tensor shape analysis over the IR
- Spec ingestion from `@requires`/`@ensures` decorators
- Implicit safety conditions (bounds checking, shape compatibility)
- Unsupported construct warnings (try/except, generators, walrus, f-strings, mutations, recursion) with strict mode
- 16 tests covering the full pipeline

#### Phase 2 — Proof Environment
- **Proof state** — full game state representation (open goals, hypotheses, tactic history)
- **Tactic action space** — 39 curated tactics across 10 categories (intro, apply, simp, omega, ring, cases, induction, etc.)
- **Lean 4 server manager** — JSON-RPC 2.0 subprocess communication (works on Windows and Unix)
- **Phase 1 → Phase 2 bridge** — converts spec obligations to proof states and Lean 4 theorem skeletons
- **Heuristic tactic suggestions** — goal-type-aware tactic recommendations
- **Lemma Database** — searchable embedding index of ~60 seed lemmas with n-gram hash retrieval, cosine similarity search, category filtering, and hit/miss feedback tracking
- **46 of 48 tests passing**

#### Phase 3 — RL Agent
- **State encoder** — transforms proof states into 256-dim fixed-size feature vectors
- **Policy + Value networks** — 2-layer MLP with He-initialized weights and full backpropagation through all layers
- **MCTS tree search** — UCB-guided search with Dirichlet noise for exploration
- **Self-play training loop** — generates experience, accumulates gradients, trains across iterations
- **Cold-start dataset loaders** — parsers for miniF2F, ProofNet, LeanDojo, and Lean Workbook formats with streaming JSON array reader (handles 90+ MB files)
- **Built-in seed data generator** — uses heuristic tactic suggestions to bootstrap the replay buffer
- **Supervised pre-training** — behavioural cloning on dataset-derived traces before self-play
- **Checkpointing** — save/load networks and seed data to JSON
- **42 tests** covering training convergence, MCTS correctness, deterministic simulation, dataset parsing, and cold-start pipeline

#### Phase 4 — Python → Lean Compiler
- **IR → Lean 4 translation** — deterministic compilation of NormalizedIR + SpecCollection into complete Lean 4 modules
- **Proof hole filling** — difficulty-based classification (TRIVIAL → `rfl`, EASY → `simp`/`omega`, MEDIUM → `induction`/`cases`, HARD → `sorry`)
- **Expression translation** — Python expressions, types, operators, and function calls mapped to Lean 4 equivalents
- **Benchmark suite** — 13 problems across 5 difficulty levels (pure arithmetic → tensor shapes)
- **End-to-end `compile()` entry point** — Python source → complete `.lean` file in one call
- **16 tests** covering all compiler features

#### Phase 5 — Bootstrapping & Integration (In Progress)
- **Dataset downloader** — `download_datasets.py` downloads all four external datasets (miniF2F, ProofNet, Lean Workbook, LeanDojo)
- **Training orchestrator** — `train_rl.py` provides a CLI for supervised pre-training + self-play with dataset integration
- **Lean 4 real-tactic simulator** — optional `--real-lean` flag to use a live Lean 4 server for accurate tactic simulation
- **Web application** — Flask-based browser interface with syntax-highlighted Lean output, sample selector, error display, keyboard shortcuts, and one-click download

---

## Module Reference

### `ast_extractor/`

Parses Python source into a normalized intermediate representation.

```python
from ast_extractor import parse_source, normalize

ir = parse_source("""
def add(x: int, y: int) -> int:
    return x + y
""", "example")
ir = normalize(ir)

# ir.functions[0].signature.name == "add"
# ir.functions[0].signature.parameters[0].name == "x"
```

Key types: `NormalizedIR`, `FunctionIR`, `ClassIR`, `StatementIR`, `ExpressionIR`, `TypeAnnotation`, `TensorOpKind`, `ParserConfig`, `ParserWarning`, `ParserError`

### `abstract_interpreter/`

Performs symbolic type inference and tensor shape analysis.

```python
from ast_extractor import parse_source, normalize
from abstract_interpreter import analyze

ir = parse_source(source, "example")
ir = normalize(ir)
state = analyze(ir)

# state.function_envs["add"]["x"] — inferred type
# state.shape_facts — symbolic shape constraints
# state.type_constraints — type compatibility facts
```

Key types: `AbstractState`, `AbstractValue`, `TypeDomain`, `TensorShape`, `ShapeDimension`

### `spec_ingestion/`

Extracts proof obligations from decorated Python specifications.

```python
from ast_extractor import parse_source, normalize
from spec_ingestion import extract_specs

ir = parse_source(source, "example")
ir = normalize(ir)
specs = extract_specs(ir)

# specs.total_count — number of obligations found
# specs.preconditions — list of @requires obligations
# specs.postconditions — list of @ensures obligations
```

Key types: `SpecCollection`, `ProofObligation`, `ObligationKind`, `ObligationStatus`

### `proof_engine/`

The game engine — proof state management, tactic execution, lemma database, and Lean 4 integration.

```python
from proof_engine import (
    ProofState, Goal, Hypothesis, GoalStatus,
    TacticExecutor, CORE_TACTICS,
    build_proof_state, obligation_to_lean_theorem,
    LeanEnv, suggest_tactics_for_goal,
    LemmaDatabase, LemmaSuggestion,
)

# Build a proof state from a ProofObligation (from spec_ingestion)
state = build_proof_state(obligation)

# Get heuristic tactic suggestions
suggestions = suggest_tactics_for_goal(state.open_goals[0])

# Search the lemma database for relevant lemmas
db = LemmaDatabase()
lemma = db.suggest_for_goal("x + 0 = x", tactic="rw")
# → LemmaSuggestion(lemma_name="add_zero", tactic="rw", rendered="rw [add_zero]")

# Generate a Lean 4 theorem skeleton
lean_code = obligation_to_lean_theorem(obligation)
# => produces: theorem ... : predicate := sorry

# Start a Lean 4 server (requires Lean 4 installed)
env = LeanEnv()
env.start()
executor = TacticExecutor(env)
result = executor.apply("intro x")
```

Key types: `ProofState`, `Goal`, `Hypothesis`, `GoalStatus`, `TacticStep`, `LemmaDatabase`, `LemmaEntry`, `LemmaSuggestion`
Key tactics: `TacticExecutor`, `TacticTemplate`, `TacticResult`, `CORE_TACTICS` (39 tactics across 10 categories), `ZERO_ARG_TACTICS`
Key builder: `obligation_to_goal`, `build_proof_state`, `build_proof_state_collection`, `build_proof_state_grouped_by_function`, `obligation_to_lean_theorem`, `to_proof_state`

### `lean_compiler/`

Translates Python source (via the existing IR pipeline) into complete Lean 4 modules.

```python
from lean_compiler import compile, IRToLeanCompiler, HoleFiller, BenchmarkSuite

# End-to-end compilation
source = """
@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x
"""

lean_code = compile(source, "absolute_example")
# => /- Auto-generated by Axiom Zero from module: absolute_example -/
#    import Mathlib
#    open Classical
#    theorem absolute_PRECONDITION_... (x : N) : x > 0 :=
#      by
#        simp
#    theorem absolute_POSTCONDITION_... (x : N) : result > 0 :=
#      by
#        simp

print(lean_code)

# Or use the compiler directly for more control
compiler = IRToLeanCompiler()
module = compiler.compile_module(ir, specs, state, "my_module")

# Fill proof holes by difficulty classification
filler = HoleFiller()
filled = filler.fill_all_holes(module, specs)
print(f"Filled {filler.fill_rate:.0%} of holes")

# Run the benchmark suite
suite = BenchmarkSuite()
results = suite.run_all(compile, "eval")
print(suite.summary(results))
```

Key types: `IRToLeanCompiler`, `HoleFiller`, `HoleDifficulty`, `BenchmarkSuite`, `BenchmarkProblem`, `BenchmarkResult`
Key functions: `compile()` — end-to-end Python source → Lean 4 module string

### `rl_agent/`

AlphaZero-style reinforcement learning agent for proof automation.

```python
from rl_agent import (
    PolicyValueNet, MCTS, SelfPlayTrainer, TrainingConfig,
    encode, FEATURE_DIM,
    TacticSimulator, run_episode, train_from_source,
)

# Encode a proof state as a feature vector
state = build_proof_state(obligation)
vec = encode(state.to_observation())  # 256-dim float vector

# Create policy-value network
net = PolicyValueNet()
priors, value = net.forward(vec)

# Run MCTS tree search
mcts = MCTS(net, num_simulations=50)
action_probs, root_value = mcts.search(state)
tactic = mcts.best_action(state)

# Full training pipeline from Python source
from rl_agent import train_from_source
net, stats = train_from_source(SIMPLE_SOURCE)

# Or use the SelfPlayTrainer directly
trainer = SelfPlayTrainer(proof_states)
net = trainer.run()
```

#### Dataset Loaders

```python
from rl_agent.dataset_loaders import (
    parse_leandojo_trace,
    parse_minif2f,
    parse_proofnet,
    parse_lean_workbook,
    load_dataset,
    generate_builtin_seed_data,
    convert_to_training_examples,
    save_seed_data, load_seed_data,
    DatasetFormat, RawProofTrace, RawProofStep,
)

# Load external datasets
traces = parse_minif2f("datasets/minif2f/minif2f_valid.json")
traces = parse_proofnet("datasets/proofnet/proofnet.jsonl")
traces = parse_lean_workbook("datasets/lean_workbook/lean_workbook.json")

# Or generate from built-in benchmarks
traces = generate_builtin_seed_data(proof_states, max_traces=30)

# Convert to training examples for the replay buffer
examples = convert_to_training_examples(traces)
```

Key types: `PolicyValueNet`, `MCTS`, `MCTSNode`, `SelfPlayTrainer`, `TrainingConfig`, `TrainingExample`, `EpisodeResult`, `TacticSimulator`, `RealTacticSimulator`
Key functions: `encode()`, `run_episode()`, `train_from_source()`

### `download_datasets.py`

Downloads external Lean proof datasets for cold-start pre-training.

```bash
python download_datasets.py                          # Download all datasets
python download_datasets.py --dataset minif2f,proofnet
python download_datasets.py --list-sources           # Show URLs only
```

Supports: LeanDojo (Zenodo), miniF2F (GitHub), ProofNet (GitHub), Lean Workbook (Hugging Face)

### `train_rl.py`

Orchestrates RL agent training with dataset integration.

```bash
python train_rl.py                                    # Heuristic simulator (all datasets)
python train_rl.py --datasets minif2f,proofnet        # Subset of datasets
python train_rl.py --quick                            # Quick smoke test
python train_rl.py --real-lean                        # Real Lean 4 server backend
python train_rl.py --iterations 20 --episodes 5       # Custom training params
```

### Web Application (`app.py`)

Flask-based browser interface for the Python → Lean 4 compiler.

```bash
pip install flask
python app.py                                         # Starts on http://127.0.0.1:5000
```

Features: syntax highlighting, sample programs, error display, line numbers, keyboard shortcuts (Ctrl+Enter), clipboard copy, .lean download, responsive design.

---

## Prerequisites

- **Python 3.10+**
- **(Optional) Flask** — for the web application (`pip install -r requirements.txt`)
- **(Optional) Lean 4** — for full proof verification with live server

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/axiom-zero.git
cd axiom-zero

# Install optional web dependencies
pip install -r requirements.txt

# Run the tests (no dependencies required)
python test_phase1.py
python test_phase2.py
python test_phase3.py
python test_phase4.py

# (Optional) Download external datasets for RL pre-training
python download_datasets.py --dataset minif2f,proofnet

# (Optional) Install Lean 4
# See: https://leanprover.github.io/

# (Optional) Start the web app
python app.py
```

No external Python packages are required for the core pipeline. All core modules use only the Python standard library.

---

## Usage

### Quick Start

```python
from ast_extractor import parse_source, normalize
from abstract_interpreter import analyze
from spec_ingestion import extract_specs
from proof_engine import build_proof_state_collection, obligation_to_lean_theorem

# 1. Write some Python code with specs
source = """
@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x
"""

# 2. Parse and analyze
ir = parse_source(source, "example")
ir = normalize(ir)
state = analyze(ir)

# 3. Extract proof obligations
specs = extract_specs(ir, state)
print(f"Found {specs.total_count} proof obligations")

# 4. Build proof states for the RL agent
proof_states = build_proof_state_collection(specs)

# 5. Generate Lean 4 theorem skeletons
for ob in specs.all:
    print(obligation_to_lean_theorem(ob))
    # theorem absolute_precondition (x : int) : x > 0 :=
    #   sorry
    #
    # theorem absolute_postcondition (x : int) : result > 0 :=
    #   sorry
```

### Full Pipeline Test

```bash
python test_phase1.py   # Phase 1: Parser Pipeline (16 tests)
python test_phase2.py   # Phase 2: Proof Environment (48 tests)
python test_phase3.py   # Phase 3: RL Agent (42 tests)
python test_phase4.py   # Phase 4: Python → Lean Compiler (16 tests)
```

### Web App

```bash
pip install flask
python app.py
# Open http://127.0.0.1:5000
```

---

## Project Structure

```
axiom-zero/
├── ast_extractor/           # Phase 1: Python → IR
│   ├── ir.py                #   IR data structures
│   ├── parser.py            #   Python source parser
│   ├── normalizer.py        #   IR normalization & SSA
│   └── __init__.py
├── abstract_interpreter/    # Phase 1: Type/Shape analysis
│   ├── abstract_domain.py   #   Lattice structures
│   ├── type_inference.py    #   Type inference engine
│   ├── shape_analysis.py    #   Tensor shape analysis
│   ├── interpreter.py       #   Main interpreter
│   └── __init__.py
├── spec_ingestion/          # Phase 1: Spec parsing
│   ├── obligations.py       #   Proof obligation types
│   ├── parser.py            #   Spec decorator parser
│   └── __init__.py
├── proof_engine/            # Phase 2: Proof environment
│   ├── proof_state.py       #   Game state representation
│   ├── tactics.py           #   Tactic action space
│   ├── lean_env.py          #   Lean 4 server manager
│   ├── lemma_db.py          #   Lemma database with embeddings
│   ├── builder.py           #   Phase 1→2 bridge
│   └── __init__.py
├── rl_agent/                # Phase 3: RL agent
│   ├── encoder.py           #   Proof state → feature vectors
│   ├── networks.py          #   Policy + value networks
│   ├── mcts.py              #   MCTS tree search
│   ├── self_play.py         #   Self-play training loop
│   ├── dataset_loaders.py   #   External dataset parsers
│   └── __init__.py
├── lean_compiler/           # Phase 4: Python → Lean
│   ├── ir_to_lean.py        #   IR → Lean 4 translation
│   ├── hole_filler.py       #   Proof hole classification & filling
│   ├── benchmark_suite.py   #   Benchmark problems & runner
│   └── __init__.py
├── static/                  # Web app frontend
│   ├── index.html           #   Main page
│   ├── app.js               #   Client-side logic
│   └── style.css            #   Dark theme styles
├── datasets/                # External datasets (downloaded)
│   ├── minif2f/             #   miniF2F benchmark
│   ├── proofnet/            #   ProofNet benchmark
│   └── lean_workbook/       #   Lean Workbook corpus
├── test_phase1.py           # Phase 1 tests (16)
├── test_phase2.py           # Phase 2 tests (48)
├── test_phase3.py           # Phase 3 tests (42)
├── test_phase4.py           # Phase 4 tests (16)
├── download_datasets.py     # Dataset downloader
├── train_rl.py              # Training orchestrator
├── app.py                   # Flask web server
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Roadmap

### ✅ Phase 1 — Parser Pipeline (Complete)
- Python source → normalized IR
- Abstract interpretation (type/shape analysis)
- Spec ingestion from decorators
- Unsupported construct warnings with strict mode

### ✅ Phase 2 — Proof Environment (Complete)
- Proof state game engine
- 39 curated tactics across 10 categories
- Lean 4 JSON-RPC server interface
- Phase 1 → Phase 2 bridge
- Lemma database with embedding-based retrieval (60+ seed lemmas)
- Heuristic tactic suggestions with goal-type awareness

### ✅ Phase 3 — RL Agent (Complete)
- State encoder (proof state → 256-dim feature vectors)
- Policy + value networks with full backprop
- MCTS tree search with deterministic simulation
- Self-play training loop with batch gradient accumulation
- Dataset loaders (miniF2F, ProofNet, LeanDojo, Lean Workbook)
- Cold-start seed data generation and supervised pre-training
- Checkpointing and persistence

### ✅ Phase 4 — Python → Lean Compiler (Complete)
- IR → Lean 4 skeleton translation
- Proof hole classification and filling
- Expression and type translation (Python → Lean)
- Benchmark suite with 13 problems across 5 difficulty levels

### 🚧 Phase 5 — Bootstrapping & Scaling (In Progress)
- ✅ Dataset downloader (miniF2F, ProofNet, Lean Workbook, LeanDojo)
- ✅ Training orchestrator with CLI (`train_rl.py`)
- ✅ Real Lean 4 server integration (`--real-lean` flag)
- ✅ Web application UI (Flask)
- 📋 Train the RL agent on the full benchmark suite
- 📋 Scale to larger Python programs and PyTorch models
- 📋 Proof verification against live Lean 4 kernel

---

## Key Design Decisions

**No external dependencies for core pipeline.** The AST extractor, abstract interpreter, proof engine, and RL agent use only Python's standard library (`ast`, `dataclasses`, `enum`, `typing`, `json`, `math`). Only the web app requires Flask.

**Lean 4 as the oracle.** By using the real Lean 4 kernel as the verifier, there is no "approximately correct." A proof either compiles or it doesn't, providing a clean binary reward signal for RL training.

**AlphaZero-style self-play.** The agent generates its own training data through self-play, with the Lean kernel providing the +1/0 reward. No human-labelled proofs are needed.

**Cold-start with external datasets.** Before self-play begins, the agent can learn from human-written proofs via supervised behavioural cloning on miniF2F, ProofNet, LeanDojo, and Lean Workbook.

**Curated tactic space.** Starting with 39 tactics across 10 categories, covering the common patterns found in simple-to-intermediate proofs. The tactic space can be extended as the agent improves.

**Lemma database with n-gram embeddings.** Instead of hardcoding lemma lookups, the lemma database uses character-level trigram hashing and cosine similarity to suggest relevant lemmas for any goal type.

---

## License

MIT
