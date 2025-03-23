import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import Tool
from crewai_tools import FileReadTool, FileWriteTool
from tools.data_tools import DataFrameLoadTool, DataFrameAnalysisTool, DataVisualizationTool

# Load environment variables from .env file
load_dotenv()

# Ensure you have set OPENAI_API_KEY in your environment or .env file
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file or environment.")

class DataAnalysisCrew:
    def __init__(self):
        # Initialize tools
        self.file_reader = FileReadTool()
        self.file_writer = FileWriteTool()
        self.df_loader = DataFrameLoadTool()
        self.df_analyzer = DataFrameAnalysisTool()
        self.visualizer = DataVisualizationTool()
        
    def create_agents(self):
        # Data Engineer Agent - Loads and prepares datasets
        data_engineer = Agent(
            role="Data Engineer",
            goal="Load, clean, and prepare datasets for analysis",
            backstory="""You are an expert data engineer skilled in loading and preparing data 
            from various formats (CSV, Excel, etc.). You're meticulous about data quality 
            and ensure datasets are properly cleaned before analysis.""",
            verbose=True,
            tools=[
                Tool.from_function(
                    func=self.file_reader.run,
                    name="FileReader",
                    description="Read the contents of a file"
                ),
                Tool.from_function(
                    func=self.df_loader.run,
                    name="DataFrameLoader",
                    description="Load a dataset from a file path into a pandas DataFrame and return basic information"
                )
            ]
        )
        
        # Data Analyst Agent - Performs the main analysis
        data_analyst = Agent(
            role="Data Analyst",
            goal="Analyze datasets and extract valuable insights",
            backstory="""You are a skilled data analyst with expertise in statistical analysis,
            data exploration, and pattern recognition. You have years of experience working with
            various datasets and can quickly identify important trends and insights.""",
            verbose=True,
            tools=[
                Tool.from_function(
                    func=self.df_analyzer.run,
                    name="DataFrameAnalyzer",
                    description="Analyze a DataFrame using pandas operations and return insights"
                ),
                Tool.from_function(
                    func=self.visualizer.run,
                    name="DataVisualizer",
                    description="Create visualizations from DataFrame data and save them to the specified path"
                )
            ]
        )
        
        # Research Specialist - Ensures proper citations and context
        research_specialist = Agent(
            role="Research Specialist",
            goal="Provide context and proper citations for data analysis",
            backstory="""You are a research specialist with a background in academic research.
            You ensure that all information is properly cited and contextually accurate.
            You're skilled at connecting data insights with their authoritative sources.""",
            verbose=True,
            tools=[
                Tool.from_function(
                    func=self.file_reader.run,
                    name="FileReader",
                    description="Read the contents of a file"
                ),
                Tool.from_function(
                    func=self.file_writer.run,
                    name="FileWriter",
                    description="Write content to a file"
                )
            ]
        )
        
        return [data_engineer, data_analyst, research_specialist]
    
    def create_tasks(self, data_source, analysis_question):
        agents = self.create_agents()
        data_engineer, data_analyst, research_specialist = agents
        
        # Task 1: Load and prepare the dataset
        data_preparation_task = Task(
            description=f"""
            Load and prepare the dataset from {data_source} for analysis.
            1. Identify the file format and load it appropriately
            2. Perform basic data cleaning (handle missing values, etc.)
            3. Provide a summary of the dataset (shape, columns, data types)
            4. Document the source of the dataset for proper citation
            
            Return the prepared dataset information and a brief summary.
            """,
            expected_output="A report on the dataset loaded, its structure, and preparation steps taken.",
            agent=data_engineer
        )
        
        # Task 2: Analyze the dataset based on the specific question
        data_analysis_task = Task(
            description=f"""
            Using the prepared dataset, analyze it to answer the following question:
            "{analysis_question}"
            
            1. Perform the necessary operations and calculations
            2. Create relevant visualizations if needed
            3. Clearly explain your findings
            4. Identify any limitations in the data or analysis
            
            Return comprehensive analysis results with visualizations.
            """,
            expected_output="Detailed analysis results that answer the specific question asked.",
            agent=data_analyst
        )
        
        # Task 3: Compile final report with proper citations
        final_report_task = Task(
            description=f"""
            Create a comprehensive final report based on the previous analyses.
            The report should:
            
            1. Introduce the dataset and its source
            2. Summarize the data preparation process
            3. Present the analysis results in a clear, understandable way
            4. Include proper citations for all data sources
            5. Discuss any limitations and potential further analyses
            
            The final report should be well-structured, professional, and include all necessary citations.
            """,
            expected_output="A comprehensive report document with analysis results and proper citations.",
            agent=research_specialist
        )
        
        return [data_preparation_task, data_analysis_task, final_report_task]
    
    def run_crew(self, data_source, analysis_question):
        """
        Run the Data Analysis Crew to process and analyze the specified dataset.
        
        Args:
            data_source (str): Path to the dataset file
            analysis_question (str): The specific question to analyze in the dataset
            
        Returns:
            str: Path to the final report
        """
        agents = self.create_agents()
        tasks = self.create_tasks(data_source, analysis_question)
        
        # Create the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=2,
            process=Process.sequential
        )
        
        # Execute the crew's tasks
        result = crew.kickoff()
        
        # Write results to file
        output_path = "final_analysis_report.md"
        with open(output_path, "w") as f:
            f.write(result)
            
        print(f"Analysis complete! Report saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Example usage
    data_source = "datasets/sample_data.csv"  # Path to your dataset
    analysis_question = "What are the main trends in this dataset and how do they correlate with each other?"
    
    crew = DataAnalysisCrew()
    report_path = crew.run_crew(data_source, analysis_question)
    
    print(f"Report generated at: {report_path}")