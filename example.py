from main import DataAnalysisCrew
import argparse

def main():
    parser = argparse.ArgumentParser(description='Analyze a dataset using CrewAI agents.')
    parser.add_argument('--data', required=True, help='Path to the dataset file')
    parser.add_argument('--question', required=True, help='Analysis question to answer')
    parser.add_argument('--output', default='final_analysis_report.md', help='Path to save the output report')
    
    args = parser.parse_args()
    
    print(f"Initializing Data Analysis Crew...")
    crew = DataAnalysisCrew()
    
    print(f"Starting analysis of dataset: {args.data}")
    print(f"Question to answer: {args.question}")
    
    report_path = crew.run_crew(args.data, args.question)
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {report_path}")
    
    # Display the first few lines of the report
    with open(report_path, 'r') as f:
        print("\nReport Preview:")
        print("-" * 40)
        preview_lines = f.readlines()[:10]
        for line in preview_lines:
            print(line.strip())
        print("-" * 40)
        print(f"See full report at {report_path}")

if __name__ == "__main__":
    main()