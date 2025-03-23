# DataAnalysis CrewAI

A Crew.AI-based system for analyzing public datasets, answering questions, and providing properly cited insights.

## Features

- Load data from various formats (CSV, Excel, JSON)
- Perform comprehensive data analysis
- Generate visualizations
- Create detailed reports with proper citations
- Multi-agent system with specialized roles

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/dataanalysis-crewai.git
cd dataanalysis-crewai
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Basic Usage

Run the example script:

```bash
python example.py --data path/to/your/dataset.csv --question "What are the main trends in this dataset?"
```

### Custom Analysis

You can also use the DataAnalysisCrew class directly in your own scripts:

```python
from main import DataAnalysisCrew

# Initialize the crew
crew = DataAnalysisCrew()

# Run analysis
report_path = crew.run_crew(
    data_source="path/to/dataset.csv",
    analysis_question="What is the correlation between X and Y?"
)

print(f"Report saved to: {report_path}")
```

## Example Datasets

You can test this system with public datasets from sources like:

1. [Kaggle Datasets](https://www.kaggle.com/datasets)
2. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
3. [Data.gov](https://data.gov/)
4. [World Bank Open Data](https://data.worldbank.org/)
5. [Google Dataset Search](https://datasetsearch.research.google.com/)

## How It Works

The system uses three specialized AI agents:

1. **Data Engineer** - Loads and prepares the dataset
2. **Data Analyst** - Analyzes the data and creates visualizations
3. **Research Specialist** - Compiles the final report with proper citations

These agents work sequentially to process your dataset and answer your questions with thorough analysis and proper attribution.

## Customization

You can customize the agents, tasks, and tools to fit your specific needs by modifying the relevant classes in the codebase.

## License

MIT