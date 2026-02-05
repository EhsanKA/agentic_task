# Agentic Task - Gemini 3 Pro Benchmark

> **Note:** Use Google Colab environment. You must use Google Colab Pro to enable Gemini 3 as an agent. All submitted notebooks must run end-to-end without manual intervention.

---

## Question 1: Agentic Task (Gemini 3)

### Objective

Design an evaluation task that meaningfully tests the headroom of Gemini-3-Pro on agentic, data-driven reasoning. The task must be headroom for Gemini-3-pro.

---

## Requirements

You must create one complete benchmark question that satisfies all of the following:

### 1. Data-Driven

- The task must require interaction with real or synthetic datasets (e.g., CSV, JSON, Parquet)
- The answer cannot be produced without reading and processing the data
- The dataset must be included or clearly specified

### 2. Agentic (Multi-Step, Environment-Interactive)

- The task must **not** be solvable via a single or a few isolated LLM calls
- It must require:
  - Multiple reasoning steps
  - Interaction with the execution environment (e.g., file inspection, validation, artifact generation)
  - Intermediate state tracking or verification
- The task should align with agentic prompt design principles

### 3. Unit Tests (Full Coverage)

You must write comprehensive unit tests that:

- Verify all required agent actions
- Validate output structure, correctness, and constraints
- Catch common failure modes
- Tests must be **deterministic**

### 4. Golden Solution (Separate Colab)

- Provide a separate Colab notebook containing a golden solution
- The golden solution must demonstrate that:
  - The task is solvable
  - The task does not rely on hidden information
- The golden solution must pass all unit tests

### 5. Execution Constraints

- The solution must run on **Google Colab Pro**
- The agent configuration must explicitly use **Gemini-3-Pro**

---

## Deliverables for Question 1

| Deliverable | Description |
|-------------|-------------|
| Benchmark question prompt | The complete task specification for the agent |
| Agent Colab | Notebook with agent setup and execution |
| Dataset | Dataset files or dataset generation code |
| Golden solution notebook | Separate notebook with reference solution |
| Unit tests | Included in **both** Golden solution and Agent Colab in clearly separated sections |

---
