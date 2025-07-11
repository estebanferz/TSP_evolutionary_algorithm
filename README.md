# TSP Evolutionary Algorithm

This project implements an evolutionary (genetic) algorithm to solve the **Traveling Salesman Problem (TSP)**.

## Description

The algorithm aims to find the shortest possible route that visits each city exactly once and returns to the starting point. It uses various evolutionary techniques, including:

- Permutation-based representation
- Selection operators (Tournament, Roulette)
- Crossover operators (PMX, OX)
- Mutation operators (Swap, Inversion)
- Survivor Selection operators (Elitism and SteadyState)

---

## Requirements

- Python 3.9 or higher
- Dependencies listed in `requirements.txt`

---

## Create a Python Virtual Environment

It’s recommended to use a virtual environment to manage project dependencies. Follow the steps below based on your operating system:

### Windows

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

### MacOS/Linux

```bash
python3 -m venv venv

source venv/bin/activate

pip3 install -r requirements.txt
```
