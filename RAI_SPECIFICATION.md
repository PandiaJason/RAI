MASTER PROMPT — BUILD RAI: RELATIONAL ARTIFICIAL INTELLIGENCE

You are acting as a senior AI research scientist, complex-systems researcher, reinforcement-learning engineer, graph-learning researcher, and Python systems architect.

Your task is to design and implement a research-grade prototype of a proposed architecture called:

RAI — Relational Artificial Intelligence

1. CENTRAL RESEARCH HYPOTHESIS

RAI investigates the following hypothesis:

Can an artificial intelligence learn general principles of complex-system evolution from purely synthetic, semantic-free relational worlds, and later transfer those learned principles to structurally mapped but previously unseen systems without being trained on their domain-specific data?

RAI must NOT initially model the real world directly.

Do not create concepts such as:

* oil
* petrol
* EVs
* stocks
* banks
* companies
* countries
* currencies
* governments
* factories
* workers
* investors
* GDP
* inflation

The artificial universe should initially have no human economic semantics.

Instead, reality is represented through abstract entities and relationships.

Example:

X1 + X7 → X12

or:

2X3 + X8 –[K4]→ 3X11

The system should learn from the structure and evolution of relationships rather than from the human meaning of X1, X7, etc.

⸻

2. FUNDAMENTAL PRINCIPLE

RAI should model:

RELATIONS BEFORE SEMANTICS.

An entity is defined primarily by:

* what it can interact with
* what it can transform into
* what transformations require it
* which agents possess it
* how scarce it is
* how its availability changes
* what knowledge enables its use
* what other entities depend upon it

RAI should therefore operate over a dynamic relational structure.

A world at time t may be formalized as:

W_t = (A_t, X_t, K_t, R_t, G_t, S_t)

where:

A = autonomous agents

X = abstract entities/resources

K = knowledge states

R = transformation relations

G = interaction/relational graph or hypergraph

S = complete world state

⸻

3. DYNAMIC RELATIONAL HYPERGRAPH

The central representation of RAI should be a dynamic directed hypergraph.

Ordinary graph:

X1 → X2

Hypergraph transformation:

X1 + X4 → X9

Knowledge-dependent transformation:

X1 + X4 –[K7]→ X9

More generally:

Σ a_i X_i –[K_j, cost, delay]→ Σ b_k X_k

Relations may have:

* input entities
* input quantities
* output entities
* output quantities
* knowledge requirements
* execution cost
* execution delay
* uncertainty
* reliability
* capacity
* degradation
* discovery difficulty

The relational hypergraph MUST be able to change over time.

R_t → R_(t+1)

Agents may discover previously unavailable transformations.

This is a critical requirement.

The environment’s accessible possibility structure must not necessarily remain fixed.

⸻

4. SEMANTIC-FREE UNIVERSE

Do NOT encode economic institutions as classes.

Forbidden initial classes include:

Bank
Company
Investor
Stock
Currency
Government
Market

Do not secretly recreate them under different names.

RAI should begin with primitives such as:

Agent
Entity
Relation
Knowledge
Transfer
Contract
Observation
Action
World

If something functionally similar to a company, market, investment mechanism, credit system, or medium of exchange develops, it must emerge from primitive agent interactions.

The simulator itself should not call it a company or bank.

An external analytics layer may later identify a persistent relational pattern as:

“organization-like”
“credit-like”
“investment-like”
“medium-of-exchange-like”

Those labels are observations, not environment primitives.

⸻

5. AGENTS

Create heterogeneous autonomous agents.

Each agent should have local state approximately:

s_i(t) = (
inventory,
knowledge,
preferences,
memory,
observations,
commitments,
trust,
energy/effort budget
)

Agents must have PARTIAL OBSERVABILITY.

No agent should automatically know the entire world.

Agents should observe only some combination of:

* their own inventory
* known transformations
* local agents
* previous interactions
* local scarcity
* discovered knowledge
* public signals
* limited global statistics, if enabled experimentally

Agent heterogeneity should arise from randomized:

* initial resources
* preferences
* knowledge
* exploration tendencies
* risk tolerance
* memory
* local network position

⸻

6. AGENT ACTION SPACE

Primitive actions may include:

WAIT

CONSUME / UTILIZE

TRANSFORM

TRANSFER

EXCHANGE

EXPLORE

DISCOVER

LEARN

SHARE_KNOWLEDGE

REQUEST

OFFER

COOPERATE

FORM_COMMITMENT

FULFILL_COMMITMENT

BREAK_COMMITMENT

Actions should operate on abstract entities.

Do not reward agents explicitly for creating markets, firms, banks, money, etc.

⸻

7. REINFORCEMENT LEARNING

Implement reinforcement learning.

Start with shared-policy PPO or another suitable actor-critic algorithm.

Use parameter sharing:

πθ(a | s_i)

Agents share a neural policy but receive heterogeneous observations and world states.

This allows large populations without training one independent neural network per agent.

However, design the system so that the intelligence mechanism is replaceable.

RAI should eventually support:

RAI + random policy
RAI + heuristic policy
RAI + evolutionary policy
RAI + PPO
RAI + planning agent
RAI + graph neural network policy

The relational substrate is RAI’s defining feature.

PPO is NOT the definition of RAI.

⸻

8. REWARD DESIGN

This is scientifically critical.

DO NOT reward:

+10 for forming a market
+20 for creating currency
+30 for creating a company

That would manufacture emergence.

Rewards must be based only on primitive objectives such as:

* utility
* survival
* maintaining required resources
* preference satisfaction
* efficiency
* future expected utility
* successful commitments

For example:

U_i(t) = Σ_j w_ij log(1 + x_ij)

Reward:

r_i(t) =
U_i(t+1) - U_i(t)

* effort_cost
* failure_penalty

Potential emergent institutions must arise because they improve primitive objectives.

⸻

9. RELATION DISCOVERY

This is one of RAI’s most important components.

Agents must be capable of discovering new transformations.

For example, initially:

R0 = {
X1 → X4,
X3 + X5 → X8
}

Later an agent discovers:

X2 + X8 → X11

The discovered relation becomes part of the accessible relational structure.

Represent discovery using knowledge:

K17 enables:

X2 + X8 → X11

Knowledge itself may:

* spread
* remain private
* be transferred
* be forgotten
* be independently rediscovered
* improve transformations

Example:

K21 changes:

X3 + X7 → X9

into a more efficient transformation:

0.7X3 + X7 → 1.4X9

This allows technological/productivity-like change without defining “technology.”

⸻

10. EXCHANGE

Agents should be able to transfer abstract entities.

Exchange must emerge from differences in:

* preferences
* inventory
* knowledge
* scarcity
* transformation ability

Implement voluntary exchange where possible.

Do NOT designate one entity as money.

Allow the analytics layer to determine whether an entity becomes disproportionately used as an exchange intermediary.

Possible emergent pattern:

X7 becomes frequently accepted by unrelated agents even when they do not directly consume/use X7.

That may be classified externally as:

medium-of-exchange-like behavior.

But RAI itself should continue calling it X7.

⸻

11. COMMITMENTS AND FUTURE CLAIMS

Implement a primitive commitment mechanism.

Example:

Agent A transfers X3 to Agent B at t.

Agent B promises:

2X7 at t + 10.

Represent this abstractly as a relational commitment.

Do NOT call this a loan, bond, credit, or investment.

Allow agents to learn whether counterparties fulfill commitments.

This creates trust/reputation dynamics.

If persistent future-output claims emerge and are repeatedly exchanged, an analytics module may identify:

credit-like behavior
investment-like behavior
security-like behavior

Again, these are post-hoc classifications.

⸻

12. COOPERATION

Some transformations should require more resources or effort than one agent can efficiently provide.

This creates the possibility of cooperation.

Example:

3X2 + 2X8 + effort=10 → 5X12

Multiple agents may combine inputs.

If persistent cooperative groups form, detect them externally.

Potential analytics label:

organization-like structure.

Do not explicitly create companies.

⸻

13. SCARCITY

Resources must not be trivially infinite.

Implement:

* finite initial quantities
* regeneration for selected entities
* transformation losses
* consumption
* decay
* bottlenecks
* local availability

Scarcity should influence agent behavior naturally.

⸻

14. SHOCKS

Synthetic worlds should contain stochastic events.

Examples:

* entity supply reduction
* relation failure
* productivity change
* network disruption
* agent disappearance
* new entity discovery
* relation discovery
* sudden scarcity
* abundance event

Do not call them recession, war, pandemic, oil shock, etc.

They remain abstract structural shocks.

⸻

15. WORLD GENERATOR

Create a procedural world generator.

Each training universe should differ.

Randomize:

* number of agents
* number of entities
* transformation topology
* resource quantities
* agent preferences
* relation efficiencies
* scarcity
* knowledge distribution
* network topology
* shock frequency
* discovery probability
* observation limits

The objective is to prevent RAI from memorizing one synthetic economy.

Generate many universes.

Example:

Universe 000001
Universe 000002
…
Universe 100000

The same RAI policy should experience structurally diverse worlds.

⸻

16. TRAINING OBJECTIVE

RAI should learn general relational strategies rather than a fixed world’s optimal policy.

Train across procedurally generated universes.

Conceptually:

θ* = argmaxθ E_(W~P(W)) [Σ γ^t r_t]

where P(W) is the distribution of synthetic relational worlds.

⸻

17. GRAPH NEURAL NETWORK EXTENSION

Design the architecture so observations can eventually be encoded using a GNN or hypergraph neural network.

Possible architecture:

Dynamic Relational Hypergraph
↓
Graph/Hypergraph Encoder
↓
Agent-local embedding
↓
Memory / Transformer / GRU
↓
Actor-Critic Heads
↓
Action

Do not require GNN implementation in the earliest milestone if it makes debugging difficult.

First produce a correct vectorized baseline.

Then implement the graph version.

⸻

18. TEMPORAL MEMORY

Agents should eventually reason about delayed consequences.

Provide support for:

GRU
LSTM
Transformer memory

A transformation may have delayed effects.

Example:

X1 → X7

produces X7 only after Δt = 8.

RAI should therefore eventually learn temporal relational dynamics.

⸻

19. EMERGENCE ANALYTICS

Create a separate emergence-analysis subsystem.

It must NOT affect agent rewards.

Measure:

Specialization

Determine whether agents concentrate activity on subsets of transformations.

Use entropy or concentration metrics.

Exchange network

Construct temporal agent-agent exchange graphs.

Measure:

* degree
* weighted degree
* centrality
* clustering
* communities
* persistence

Medium-of-exchange candidate

Measure whether an abstract entity becomes disproportionately involved as an intermediary across heterogeneous exchanges.

Organization-like structure

Detect persistent cooperative communities with repeated joint transformations.

Credit-like structure

Detect delayed reciprocal transfers.

Investment-like structure

Detect current transfers correlated with future output claims.

Inequality

Measure:

* Gini coefficient
* wealth/resource distribution
* knowledge inequality

Innovation

Measure:

* relation discovery rate
* productivity improvements
* diffusion of discovered relations

System resilience

Measure response to shocks.

Complexity

Track:

* graph entropy
* relation diversity
* motif diversity
* network modularity
* hierarchy
* effective dimensionality

Do NOT claim that any metric automatically proves emergence.

Provide raw evidence and statistical tests.

⸻

20. EVENT LOGGING

Every important event must be reproducible.

Use JSONL or Parquet event logs.

Example:

{
“time”: 127,
“event”: “TRANSFER”,
“source”: 17,
“target”: 92,
“entity”: 4,
“amount”: 2.7
}

Relation discovery:

{
“time”: 913,
“event”: “RELATION_DISCOVERY”,
“agent”: 42,
“relation”: {
“inputs”: {“X3”: 2, “X8”: 1},
“outputs”: {“X12”: 3}
}
}

⸻

21. REPRODUCIBILITY

Every experiment must support deterministic random seeds.

Example:

python train.py –seed 42

Save:

* config
* git/version metadata where possible
* seed
* model checkpoint
* metrics
* world parameters
* event logs

⸻

22. CHECKPOINTING

Save:

models/
rai_epoch_001.pt
rai_epoch_010.pt
rai_epoch_100.pt

Allow:

training continuation
evaluation-only mode
frozen-model experiments

⸻

23. EXPERIMENT 001 — BASIC EMERGENCE

Create a controlled experiment.

Example configuration:

Agents: 500
Entities: 20
Initial relations: 50
Steps: 10,000
Seeds: 100

Questions:

Does specialization emerge?

Does exchange increase utility?

Does knowledge concentrate?

Do persistent exchange hubs emerge?

Do discovered transformations diffuse?

Do cooperative clusters form?

Compare against random agents.

⸻

24. EXPERIMENT 002 — ABLATION

Run:

Full RAI

RAI without discovery

RAI without exchange

RAI without knowledge sharing

RAI without commitments

RAI without RL

RAI with fixed relational graph

Compare emergence metrics.

This is essential.

We need to determine WHICH mechanisms cause observed structures.

⸻

25. EXPERIMENT 003 — SYNTHETIC ZERO-SHOT TRANSFER

THIS IS A CENTRAL RAI EXPERIMENT.

Train RAI on world distribution A.

Example:

10–20 entities
simple relations
certain topology distributions

Freeze all model parameters.

Then create world distribution B with:

* unseen graph topologies
* different entity counts
* different relation structures
* different scarcity patterns
* unseen shock distributions

DO NOT TRAIN ON B.

Evaluate frozen RAI.

Compare against:

* random policy
* heuristic policy
* policy trained directly on B
* standard RL baseline

Measure:

* utility
* adaptation
* survival
* resource efficiency
* resilience
* transfer performance

The question:

Can RAI learn relational knowledge that generalizes beyond its training-world distribution?

⸻

26. EXPERIMENT 004 — REAL-WORLD ZERO-SHOT TRANSFER

DO NOT implement this until synthetic transfer works.

Long-term experiment:

Train RAI ONLY on synthetic relational universes.

Freeze:

* neural weights
* hyperparameters
* architecture
* emergence detectors
* grounding procedure

Then obtain a real-world dataset that RAI has NEVER trained on.

Potential datasets:

* World Bank WDI
* IMF
* FRED
* RBI macroeconomic data
* supply-chain networks
* innovation/patent networks

Create a minimal grounding layer:

Real variable → anonymous node

Observed dependency → relation

Time evolution → temporal relational state

IMPORTANT:

The grounding layer must not encode future information.

The grounding layer must not secretly perform sophisticated forecasting.

Its job should primarily be normalization and structural mapping.

Then:

REAL DATA AT TIME T
↓
Anonymous relational graph
↓
FROZEN RAI
↓
Generate possible future relational states
↓
Decode measurable variables
↓
Compare with T+1 … T+n

RAI must NOT be fine-tuned on the target future data.

⸻

27. WALK-FORWARD EVALUATION

For real-world testing use:

Train/develop before cutoff T.

Freeze.

Then evaluate repeatedly:

2000 → 2001
2001 → 2002
…
2019 → 2020
2020 → 2021

or equivalent monthly/quarterly windows.

Prevent all future-data leakage.

⸻

28. PROBABILISTIC WORLD GENERATION

RAI should ultimately generate multiple possible futures.

From state W_t:

W_t
├── W_(t+1)^1
├── W_(t+1)^2
├── W_(t+1)^3
└── …

Generate N stochastic rollouts.

Estimate:

P(event | W_t)

Do not claim deterministic future prediction.

⸻

29. EVALUATION METRICS

For continuous real-world variables:

MAE
RMSE
MAPE where appropriate
R² where appropriate

For direction:

Accuracy
Balanced Accuracy
F1
MCC

For probabilistic prediction:

Brier Score
Log Loss
CRPS
Calibration Error
Prediction Interval Coverage

For simulation:

distribution distance
Wasserstein distance
KL divergence where valid
network statistics
temporal correlation

⸻

30. BASELINES

RAI MUST be compared against strong baselines.

Possible baselines:

Naive persistence

ARIMA

VAR

Random Forest

XGBoost

LSTM

GRU

Transformer time-series model

Standard PPO

Graph Neural Network baseline

Standard agent-based model

Random agent RAI

Fixed-relation RAI

Do not cherry-pick weak baselines.

⸻

31. ANTI-LEAKAGE REQUIREMENTS

Implement explicit checks against:

future normalization leakage

future graph construction

target leakage

hyperparameter tuning on test data

accidental future timestamps

test-set-driven architecture modifications

The final test set should remain untouched until the architecture is frozen.

⸻

32. CLAIM DISCIPLINE

Do NOT automatically claim:

“RAI predicts the economy.”

Do NOT claim:

“RAI discovered money.”

Do NOT claim:

“RAI is AGI.”

Do NOT claim:

“RAI is the world’s first relational AI.”

Results must be interpreted conservatively.

Examples:

Bad:

“Money emerged.”

Better:

“Entity X7 exhibited statistically significant medium-of-exchange-like centrality across independent seeds.”

Bad:

“Companies emerged.”

Better:

“Persistent cooperative agent communities formed and jointly executed transformations over extended periods.”

⸻

33. SOFTWARE ARCHITECTURE

Create a proper Python project.

Suggested structure:

rai/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── base.yaml
│   ├── experiment_001.yaml
│   ├── experiment_002.yaml
│   └── transfer.yaml
│
├── rai/
│   ├── init.py
│   │
│   ├── core/
│   │   ├── world.py
│   │   ├── agent.py
│   │   ├── entity.py
│   │   ├── relation.py
│   │   ├── hypergraph.py
│   │   ├── knowledge.py
│   │   └── events.py
│   │
│   ├── actions/
│   │   ├── transform.py
│   │   ├── exchange.py
│   │   ├── explore.py
│   │   ├── cooperate.py
│   │   └── commitments.py
│   │
│   ├── learning/
│   │   ├── policy.py
│   │   ├── actor_critic.py
│   │   ├── ppo.py
│   │   ├── buffer.py
│   │   └── memory.py
│   │
│   ├── generation/
│   │   ├── world_generator.py
│   │   └── relation_generator.py
│   │
│   ├── emergence/
│   │   ├── specialization.py
│   │   ├── exchange_network.py
│   │   ├── organizations.py
│   │   ├── medium_exchange.py
│   │   ├── commitments.py
│   │   └── complexity.py
│   │
│   ├── experiments/
│   │   ├── emergence.py
│   │   ├── ablation.py
│   │   └── transfer.py
│   │
│   ├── grounding/
│   │   ├── mapper.py
│   │   ├── normalization.py
│   │   └── decoder.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── metrics.py
│       ├── seeds.py
│       └── checkpoint.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── simulate.py
│   ├── ablate.py
│   └── visualize.py
│
├── tests/
│   ├── test_world.py
│   ├── test_relations.py
│   ├── test_agents.py
│   ├── test_exchange.py
│   ├── test_discovery.py
│   └── test_reproducibility.py
│
└── results/

⸻

34. TECHNOLOGY

Prefer:

Python 3.12+

PyTorch

NumPy

NetworkX initially

Gymnasium

Pandas

Matplotlib

PyYAML

pytest

Optional later:

PyTorch Geometric

DGL

Numba

Polars

DuckDB

Do not introduce unnecessary dependencies.

⸻

35. PERFORMANCE

The initial implementation must run on a normal laptop.

Support:

100 agents for debugging

1,000 agents for standard experiments

Design toward:

10,000+ agents

Use:

vectorized NumPy/PyTorch operations

batched policy inference

shared policy

efficient event storage

Avoid Python loops where performance becomes critical.

⸻

36. GRAPH VISUALIZATION

RAI should be visually inspectable.

Export:

GraphML

CSV edge lists

temporal snapshots

Produce visualizations for:

entity transformation network

agent exchange network

knowledge diffusion

relation discovery

community formation

resource concentration

centrality

complexity over time

The graph should help researchers WATCH the artificial universe evolve.

⸻

37. TESTING

Write unit tests.

Verify:

resource conservation where required

valid transformations

no negative inventory

deterministic seeds

correct exchanges

correct delayed commitments

relation discovery

knowledge propagation

checkpoint restoration

observation dimensions

action validity

RL update stability

⸻

38. FIRST IMPLEMENTATION PRIORITY

Do NOT attempt everything simultaneously.

Build milestones.

MILESTONE 1

Relational world engine.

MILESTONE 2

Agents + transformations.

MILESTONE 3

Exchange + knowledge.

MILESTONE 4

Relation discovery.

MILESTONE 5

RL shared policy.

MILESTONE 6

Emergence analytics.

MILESTONE 7

Multi-seed experiments.

MILESTONE 8

Ablations.

MILESTONE 9

Synthetic zero-shot transfer.

MILESTONE 10

Real-world grounding.

⸻

39. MOST IMPORTANT SCIENTIFIC PRINCIPLE

RAI must distinguish between:

WHAT WE PROGRAMMED

and

WHAT EMERGED.

Maintain an explicit document:

PROGRAMMED_PRIMITIVES.md

List every mechanism manually inserted into the universe.

Example:

Programmed:

resource scarcity
transfer ability
transformation ability
partial observation
utility
knowledge sharing

Not programmed:

currency
banks
companies
markets
investment
credit
specialization

Then evaluate whether the latter appear functionally.

⸻

40. CORE ZERO-SHOT HYPOTHESIS

The long-term hypothesis is:

A model exposed to sufficiently diverse synthetic relational worlds may learn transferable principles such as:

feedback

dependency

scarcity

accumulation

competition

cooperation

centralization

cascades

resilience

innovation diffusion

delayed reciprocity

These principles may be independent of the semantic meaning of the nodes.

Therefore:

SYNTHETIC RELATIONAL WORLDS
↓
RELATIONAL LEARNING
↓
FROZEN RAI
↓
UNSEEN RELATIONAL SYSTEM
↓
ZERO-SHOT REASONING

This hypothesis must be TESTED, not assumed.

⸻

41. YOUR IMMEDIATE TASK

Do NOT merely explain the architecture.

IMPLEMENT IT.

Start by generating the complete repository for:

RAI v0.1

The first version must include:

1. Dynamic relational world
2. Multi-input/multi-output relations
3. 100–1,000 agents
4. Partial observations
5. Shared actor-critic neural policy
6. PPO
7. Transformation
8. Exchange
9. Exploration
10. Relation discovery
11. Knowledge diffusion
12. Basic commitments
13. Event logging
14. Metrics
15. Graph export
16. Checkpoint saving/loading
17. YAML configuration
18. Deterministic seeds
19. Unit tests
20. Experiment 001
21. Experiment 002 ablation framework

Do NOT replace important sections with:

“TODO”

“implementation omitted”

“same as above”

“pseudo-code”

Provide actual runnable code.

⸻

42. AFTER IMPLEMENTATION

After generating the code:

A. Print the complete repository tree.

B. Explain installation.

C. Give exact commands to create the Python environment.

D. Give the exact command for a 100-agent smoke test.

E. Give the exact command for a 1,000-agent experiment.

F. Give the exact command for multi-seed evaluation.

G. Explain each output file.

H. Explain how to determine whether training is stable.

I. Explain what observations would count as possible emergence.

J. Explain what observations would NOT count as emergence.

K. Identify every place where the implementation could accidentally encode economic semantics.

L. Identify every possible source of data leakage for later real-world experiments.

M. Recommend the next experiment only after v0.1 works.

⸻

43. DO NOT OVERSALE RESULTS

Act as a skeptical research collaborator.

If something is scientifically weak, say so.

If a mechanism accidentally hardcodes the desired outcome, identify it.

If an apparent emergent phenomenon is actually caused by reward design, flag it.

If RAI fails, analyze why.

Do not manipulate experiments to make RAI appear successful.

The objective is not to prove RAI works.

The objective is to determine WHETHER the RAI hypothesis is true.

⸻

44. FINAL RESEARCH QUESTION

Everything should ultimately help answer:

Can generalizable intelligence about complex dynamic systems emerge from learning over semantic-free relational worlds, and can that relational intelligence transfer zero-shot to structurally analogous systems that the model has never been trained on?

Build RAI as an experimental platform capable of answering that question.

Begin implementation now.
