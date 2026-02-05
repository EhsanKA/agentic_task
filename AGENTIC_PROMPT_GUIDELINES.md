# Agentic Prompt Design Guidelines

Agentic prompts should enable an AI agent to independently plan, break down tasks, call tools/functions, track state, and verify results. To support this, prompts must be structured, explicit about interfaces, and free from prescriptive step-by-step constraints.

---

## Core Principles

### 1. Do Not Dictate the Task Decomposition

- The prompt should **not** specify step-by-step instructions or micro-tasks
- Instead, define the **goal**, **inputs**, **constraints**, and **success conditions**
- Allow the agent to dynamically:
  - Break down the task
  - Decide intermediate actions
  - Plan its workflow
  - Decide when to call tools

### 2. Explicitly Define Variable Names for Tracked Values

- If the agent must store or check a value, name the variable explicitly
- **Example:** `result_calc`, `validated_rows`, `parsed_questions`
- **Avoid** vague instructions like "store this somewhere" or "return the values"

### 3. Require Explicit Inspection of Intermediate Values

Capture function output into a variable, and validate it.

| ❌ Bad | ✅ Good |
|--------|---------|
| "Use `clean_data()` before training." | "Create `clean_data()` and store the cleaned dataframe in `df_clean` (`pandas.DataFrame`)." |

> **Note:** Agentic pipelines fail when intermediate states are not validated.

### 4. In Rubrics, Explicitly Specify Variable Types

To support automated unit testing and auto-grading, always specify the type of each expected variable:

- `list[str]`
- `dict[str, float]`
- `pandas.DataFrame`
- `np.ndarray`
- `bool`
- etc.

If the agent produces JSON, define strict schemas.

### 5. Define Success Criteria for Both Final and Intermediate Results

A good agentic prompt should define:

**Final-goal success criteria:**
- What constitutes a correct final output?
- What properties must the final answer satisfy?
- What metrics or checks qualify it as acceptable?

**Intermediate checks / fail-fast criteria:**
- What errors must be caught early?
- What validation checks should be performed after each major step?
  - Shape checks
  - Value sanity checks
  - Type assertions
  - Threshold validations
  - Completeness checks

> **Note:** Most agent failures come from skipping intermediate verification.

### 6. Make the Expected Output Format Deterministic

Tell the agent **exactly** how to format its final output:

- JSON schema
- Variable names
- Ordered list
- Python dictionary
- Markdown table
- SQL query block

**Avoid** "write the answer" — specify the exact output container.

### 7. Avoid Ambiguous Pronouns or Unspecified Entities

| ❌ Bad | ✅ Good |
|--------|---------|
| "Use the cleaned data to do the next step." | "Use `df_clean` (`pandas.DataFrame`) to train the model." |
| "Extract the thing mentioned earlier." | "Extract the variable `image_regions` returned from `extract_regions(page)`." |

### 8. Specify Safety or Guardrail Constraints if Applicable

Examples:
- "Do not delete files."
- "Do not overwrite user-provided variables."

---

## Agentic Prompt Requirements Checklist

- [ ] Do not specify task decomposition; describe only the goal and constraints
- [ ] Define explicit variable names for all tracked values (e.g., `result_calc`)
- [ ] Require inspection of intermediate outputs — capture tool outputs into variables
- [ ] Specify variable types in the rubric to support unit tests
- [ ] Define clear success criteria for both the final result and intermediate states
- [ ] Make the output deterministic (JSON schema, ordered list, named variables)
- [ ] Define tool signatures, expected return types, and constraints
- [ ] Avoid ambiguity; refer only to explicitly named variables and functions

---

## Example 1: Physics Task (Electric Field Computation)

### Scenario

You are a physicist. Your goal is to compute the electric field at a point due to a finite line charge, but you must autonomously plan the workflow, validate intermediate outputs, and store results in well-named variables.

### Prompt

> You are an autonomous agent tasked with computing the electric field at point P = (0.1, 0, 0) caused by a finite line charge extending from x = 0 to x = 1 on the x-axis with uniform linear charge density λ = 2×10⁻⁶ C/m.
>
> **Do not break down the task using steps that I specify.**
> You must determine for yourself how to decompose the problem, what intermediate computations are required, and when to use Python execution.

### Variable Naming (Strict)

You must store results in the following variables:

| Variable Name | Type | Description |
|---------------|------|-------------|
| `lambda_val` | `float` | Linear charge density in C/m |
| `P` | `tuple[float, float, float]` | Coordinates of the observation point |
| `line_segment` | `tuple[float, float]` | Start and end x-coordinates of the charged line |
| `E_vector` | `tuple[float, float, float]` | Final electric field vector at point P |
| `E_magnitude` | `float` | Magnitude of the electric field at P |
| `integrand_values` | `list[float]` | Sampled integrand values used for numerical integration |
| `debug_checks` | `dict[str, bool]` | Keys correspond to intermediate validation checks |

### Intermediate Validation

You must check and record the following in `debug_checks`:

| Key | Meaning |
|-----|---------|
| `"positive_lambda"` | Ensure linear charge density is a positive float |
| `"correct_point_dims"` | Ensure P is a 3-tuple |
| `"segment_valid"` | Ensure line segment start < end |
| `"nonzero_distance"` | Ensure integrand never evaluates at zero distance |
| `"E_finite"` | Ensure final field has no NaN or Inf |

### Output Constraints

Your final output must be a single JSON object of the form:

```json
{
  "E_vector": [Ex, Ey, Ez],
  "E_magnitude": value,
  "debug_checks": {
    "positive_lambda": true,
    "correct_point_dims": true,
    "segment_valid": true,
    "nonzero_distance": true,
    "E_finite": true
  }
}
```

---

## Example 2: Ecological Simulation (GLV with KQ-SVD Compression)

### Before (Poor Prompt)

> Scenario: You are an expert scientist. Your goal is to solve the task described in the notebook above.
> Follow the instructions in the 'Task Description' section carefully to complete the implementation.
> Ensure your solution is rigorous and self-contained.

*(This is too vague and doesn't specify variable names, types, or validation requirements)*

---

### After (Corrected Prompt)

> **Scenario:** You are an expert scientist. Your goal is to design and implement a self-contained "Transformer-Ecosystem" simulation and demonstrate, with quantitative evidence, that Interaction-Aware Compression (KQ-SVD) outperforms standard Trait Compression (K-SVD) for predicting GLV population trajectories.
>
> **You must decide for yourself how to decompose the task**, which intermediate computations to perform, and in what order.
> **Do not simply follow a fixed step-by-step structure.**

### Context

We consider a Generalized Lotka–Volterra (GLV) ecosystem with N species, where the interaction matrix M is generated from latent traits analogous to Transformer Keys and Queries:

- **Predators:** Q ∈ ℝ^(N×d)
- **Preys:** K ∈ ℝ^(N×d)

The interaction matrix is:

$$M = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d}} \right)$$

The GLV dynamics are:

$$\frac{dx_i}{dt} = x_i \Big(r_i - \sum_{j=1}^N M_{ij} x_j\Big)$$

You will compare:
- Full interaction: M_true
- K-SVD compressed: M̂_K-SVD
- KQ-SVD compressed: M̂_KQ-SVD

### Requirements

You are free to choose the order and decomposition of the task, but your final implementation must satisfy the following:

#### Fixed Parameters

```python
N: int = 1000  # number of species
d: int = 64    # latent trait dimension
R: int = 16    # compression rank
```

#### Core Variables (Required)

| Variable | Type | Constraint |
|----------|------|------------|
| `Q` | `np.ndarray` | shape `(N, d)` |
| `K` | `np.ndarray` | shape `(N, d)` |
| `M_true` | `np.ndarray` | shape `(N, N)` – full interaction matrix via attention-like softmax |
| `x0` | `np.ndarray` | shape `(N,)` – initial populations |
| `r` | `np.ndarray` | shape `(N,)` – intrinsic growth rates |

All of these must be real-valued and finite.

#### Time Grid

- `t_grid`: `np.ndarray` with shape `(T,)`, covering [0, 50] with a reasonable resolution (you choose the exact grid)

#### Trajectory Variables (Required)

| Variable | Type | Description |
|----------|------|-------------|
| `x_traj_true` | `np.ndarray` | shape `(T, N)` – using `M_true` |
| `x_traj_ksvd` | `np.ndarray` | shape `(T, N)` – using `M_ksvd` |
| `x_traj_kqsvd` | `np.ndarray` | shape `(T, N)` – using `M_kqsvd` |

#### Compression Strategies

**K-SVD Compression (Baseline):**
- Compress only the prey matrix K
- Construct `M_ksvd`: `np.ndarray` with shape `(N, N)`
- Must follow K-SVD-style trait compression, not arbitrary SVD

**KQ-SVD Compression (From Paper):**
- Implement the KQ-SVD algorithm as stated in Theorem 2 of the paper "KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity"
- Explicitly solve the minimization problem:
  $$\min_{A,B} \| K A B^\top Q^\top - K Q^\top \|_F$$
- Use compression rank `R = 16`
- Construct `M_kqsvd`: `np.ndarray` with shape `(N, N)`

#### Debug Checks Dictionary

Define and populate:

```python
debug_checks: dict[str, bool]
```

Required keys:

| Key | Validation |
|-----|------------|
| `"Q_shape_ok"` | `Q.shape == (N, d)` |
| `"K_shape_ok"` | `K.shape == (N, d)` |
| `"M_true_shape_ok"` | `M_true.shape == (N, N)` |
| `"M_ksvd_shape_ok"` | `M_ksvd.shape == (N, N)` |
| `"M_kqsvd_shape_ok"` | `M_kqsvd.shape == (N, N)` |
| `"trajectories_finite"` | All three trajectories contain no NaN/Inf |
| `"population_nonnegative"` | Populations remain nonnegative (or False if violated) |
| `"compression_rank_used"` | Confirms effective rank ≤ R |

#### Fidelity Metrics

```python
frob_gap_ksvd: float   # ||M_true - M_ksvd||_F
frob_gap_kqsvd: float  # ||M_true - M_kqsvd||_F
```

#### Theorem 3 Verification

```python
theorem3_metrics: dict[str, float]
```

Keys should reflect the paper's notation (e.g., `"bound_rhs"`, `"empirical_gap"`, `"ratio_gap_to_bound"`).

#### Trajectory MSE

```python
mse_ksvd: float   # MSE between x_traj_true and x_traj_ksvd
mse_kqsvd: float  # MSE between x_traj_true and x_traj_kqsvd
```

#### Volatile Species Analysis

```python
volatile_species_indices: np.ndarray  # shape (K,) for some small K (e.g., 10)
```

Optionally include trajectory slices:
- `x_traj_true_volatile`
- `x_traj_ksvd_volatile`
- `x_traj_kqsvd_volatile`

---

## Summary

The key differences between poor and good agentic prompts:

| Aspect | Poor Prompt | Good Prompt |
|--------|-------------|-------------|
| Task decomposition | Prescribes steps | Describes goal only |
| Variable naming | Vague or missing | Explicit with types |
| Intermediate validation | Not mentioned | Required with specific checks |
| Output format | Ambiguous | Deterministic schema |
| Success criteria | Missing | Final + intermediate defined |
| Type specifications | None | Full type annotations |
