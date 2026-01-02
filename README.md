# Agentic Multi-Store RAG (AMS-RAG)



## Description
Agentic Multi-Store Retrieval-Augmented Generation (AMS-RAG) is a unified framework designed to overcome the limitations of "one-size-fits-all" RAG systems. While traditional RAG relies on a single vector store, real-world organizational data is heterogeneous—spanning unstructured documents, structured databases, and complex knowledge graphs.

This project implements an Orchestrator Agent that acts as an intelligent router, analysing user queries to dynamically select the most appropriate retrieval pipeline. By treating data stores as specialised tools rather than a monolithic index, AMS-RAG ensures accurate retrieval across diverse data formats.

Key Features
🤖 Orchestrator Agent: A central "brain" that classifies user intent and routes queries to the optimal pipeline (Vector, SQL, or Graph), enabling a single interface for diverse data questions.

📄 Advanced Vector Pipeline: Optimised for unstructured text and visual data (PDFs/images).

Tech Stack: Nanonets OCR for layout-aware parsing, Hybrid Retrieval (Semantic + BM25), and Cross-Encoder Re-ranking.

🗄️ SQL Pipeline: Specialised for precise lookup and aggregation of structured tabular data.

Tech Stack: Llama-3.1-8B-Instruct for Text-to-SQL, optimized with Query-Time Table Retrieval (QTTR) to focus context on relevant schemas.

🕸️ Graph Pipeline: Designed for multi-hop reasoning and relationship discovery.

Tech Stack: Text-to-Cypher generation grounded by Entity Disambiguation and Schema-Aware prompting to prevent hallucinations.

📊 Comprehensive Benchmarking: Includes a custom evaluation suite using Ragas for vector metrics and execution-based metrics (DataCompy, CypherMatch) for validating SQL and Graph query accuracy.

## 📂 Directory Structure

The repository is organized into three main directories: **Agent** for the active application, **Datasets** for data management, and **Archive** for development history and benchmarks.

```plaintext
├── Agent/                  # Unified pipeline and web interface
│   ├── app.py              # Streamlit web application
│   └── agent.py            # Agent Pipeline
│
├── Datasets/               # Data sources for RAG pipelines and evaluation
│   ├── Vector/             # Unstructured text data for the Vector pipeline
│   ├── SQL/                # Structured data (CSV files) for the SQL pipeline
│   ├── Graph/              # Knowledge graph data for the Graph pipeline
│   └── Benchmarking/       # Evaluation datasets for pipeline performance testing
│
└── Archive/                # Development artifacts, reports, and previous iterations
    ├── Base Pipelines/     # Initial implementations of Vector, SQL, and Graph pipelines
    ├── Advanced Pipelines/ # Fine-tuned versions of the base pipelines
    ├── Refactored/         # Pipelines refactored into modular Python classes
    ├── Benchmarks/         # Evaluation results for all pipeline iterations
    ├── Agent Evaluation/   # Specific performance metrics for the Agent pipeline
    └── Reports/            # Project proposals and progress reports
```

## Visuals
(Placeholder: Screenshot of Streamlit UI here showing the chat interface)

## Installation
This project requires Python 3.10+ and a Neo4j instance for the Graph pipeline.

### Setup Steps
1) Clone the repository

```plaintext
git clone https://sgts.gitlab-dedicated.com/wog/htx/htxdsaicoga/rnd/rnd-interns/heng-joo-capstone.git
cd heng-joo-capstone
```

2) Install Dependencies

```plaintext
pip install -r requirements.txt
```

3) Environment Configuration
Create a .env file in the root directory and add your necessary API keys and database credentials:

```plaintext
Example .env structure
OPENAI_API_KEY="sk-..."  # Or your local LLM endpoint
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"
NANONETS_API_KEY="..."
```

4) Database Initialization

Ensure your Neo4j instance is running.

Populate the SQL and Graph stores by initiating the ingestion process through the Streamlit Web UI (Agent/app.py) or by running the agent script directly (Agent/agent.py).

## Usage
To launch the unified Orchestrator Agent and Web Interface:

```plaintext
streamlit run Agent/app.py
```
Once running, navigate to http://localhost:8501 in your browser. You can ask questions such as:

Vector: "What were the key crime trends in 2022 based on the SPF Annual Report?"

SQL: "Calculate the total inmate population statistics from the provided CSV data."

Graph: "How is the 'HTX' entity related to 'Projects' in the organization structure?"

## Roadmap


For a detailed timeline of completed milestones and upcoming development phases, please refer to

**[📅 View Project Gantt Chart](https://docs.google.com/spreadsheets/d/1aQxM0jrjyhOpHAwQIIw-j-bsdkOsPX7a/edit?usp=sharing&ouid=103044922105855395613&rtpof=true&sd=true)**

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
