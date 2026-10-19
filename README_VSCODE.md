# 🚀 How to Run the GraphRAG Compression Trainer Locally in VS Code

This full-stack application consists of:
1. **Interactive React Frontend** and **Express Web Server** (the full web-based dashboard).
2. **Terminal-Based Python QA Trainer** (`run_qa_trainer.py`) for deep mathematical CLI playground.

Here is the exact step-by-step order to install and run both components successfully in VS Code on your local computer.

---

## 🛠️ Step 1: Install System Prerequisites

Ensure you have the following installed on your operating system:
* **Node.js** (v18 or higher recommended) -> [Download Node.js](https://nodejs.org/)
* **Python** (v3.9 or higher recommended) -> [Download Python](https://www.python.org/)
* **VS Code** (with the recommended extension "Python" and "Node" packages).

---

## 📂 Step 2: Open Project in VS Code

1. Extract your downloaded `.zip` folder.
2. In VS Code, go to **File > Open Folder...** and select the root directory of this extracted project.
3. Open VS Code's built-in terminal by pressing ``Ctrl + ` `` (or `Cmd + \`` on macOS).

---

## 📝 Step 3: Configure Environment Variables

Create a new file named `.env` in the root of the project folder (you can duplicate `.env.example`):

```bash
# Copy placeholder values to real .env
cp .env.example .env
```

Open `.env` in VS Code and fill in your keys:
```env
# REQUIRED: Your Gemini API Key for AI-driven answer grading and query grounding
GEMINI_API_KEY=your_actual_gemini_api_key_here

# OPTIONAL: Your Neo4j connection (if left empty, the system automatically runs on fallback memory database mode)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## 🛜 Step 4: Install & Launch the Full-Stack Web App (Option A)

This launches the web-based interactive GUI containing the **Interactive Graph Compressor**, **Live Metrics Panel**, and **AI Answer Evaluator**.

### 1. Install Node.js dependencies:
In your VS Code terminal, run:
```bash
npm install
```

### 2. Run the development server:
Start the Express + Vite server locally:
```bash
npm run dev
```

### 3. Open the web app:
Your terminal will output:
`[Server] Graph RAG application listening on http://0.0.0.0:3000`

Open your web browser and navigate to:
👉 **`http://localhost:3000`**

---

## 🐍 Step 5: Setup & Run Python QA CLI App (Option B)

If you wish to interact with the console terminal version of the QA trainer (`run_qa_trainer.py`):

### 1. Open a new terminal tab in VS Code.
### 2. Create and activate a Python Virtual Environment (`venv`):
* **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  call venv\Scripts\activate.bat
  ```

### 3. Install Python dependencies:
With the virtual environment active `(venv)`, run:
```bash
pip install -r requirements.txt
```

### 4. Execute the Python QA Trainer CLI:
Run the console application:
```bash
python run_qa_trainer.py
```
This launches the command-line interface where you can pick any of the 10 papers, run Tarjan compression, observe Focus and Fracture rate calculations, and verify test answers!

---

## 📊 Summary of What Runs What:
* **Web App (Vite+React Frontend + Express Server)**: Uses Node/npm. Runs on port `3000` to serve the interactive UI.
* **Python Trainer (`run_qa_trainer.py`)**: Runs directly in the terminal as a CLI. Uses Python dependencies.
* **Both apps read from your `.env` file** to retrieve your `GEMINI_API_KEY` for live AI verification!
