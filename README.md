# SampleMoa

# Mixture-of-Agents (MoA) Demo with LangChain and Ollama

This project demonstrates the Mixture-of-Agents (MoA) methodology using LangChain and Ollama. The MoA approach leverages the collective strengths of multiple large language models (LLMs) to enhance natural language understanding and generation capabilities. The script sets up multiple agents (LLMs) to iteratively refine their responses through multiple layers.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Recent advances in large language models (LLMs) have demonstrated significant capabilities in natural language understanding and generation. This project implements a Mixture-of-Agents (MoA) architecture to harness the collective expertise of multiple LLMs. Each layer comprises multiple LLM agents that take outputs from agents in the previous layer as auxiliary information to generate refined responses.

## Features

- Utilizes multiple LLMs: `llama3.1`, `qwen2`, and `mistral nemo`.
- Iterative refinement of responses through multiple layers of agents.
- Aggregates responses using semantic similarity to select the most coherent answer.
- Provides responses in Portuguese.

## Installation

1. **Clone the Repository:**

    ```sh
    git clone https://github.com/ulisses177/SampleMoa
    cd moa-demo
    ```

2. **Create and Activate a Virtual Environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Install Additional Required Packages:**

    ```sh
    pip install sentence-transformers
    ```

## Usage

To run the MoA demo, use the following command:

```sh
python moa_demo.py

