import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, Any, Optional
import base64
from io import BytesIO

class DataFrameLoadTool:
    """Tool for loading datasets into pandas DataFrames."""

    def run(self, file_path: str) -> str:
        """
        Load a dataset from a file path into a pandas DataFrame.

        Args:
            file_path (str): Path to the dataset file

        Returns:
            str: JSON string with DataFrame info and summary
        """
        try:
            # Determine file type by extension
            file_ext = file_path.split('.')[-1].lower()

            if file_ext == 'csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['xls', 'xlsx']:
                df = pd.read_excel(file_path)
            elif file_ext == 'json':
                df = pd.read_json(file_path)
            else:
                return f"Unsupported file format: {file_ext}"

            # Save the loaded DataFrame to a temporary CSV for further processing
            temp_path = "temp_dataframe.csv"
            df.to_csv(temp_path, index=False)

            # Get basic DataFrame info
            info_dict = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head(5).to_dict(orient='records'),
                "file_path": file_path,
                "temp_path": temp_path
            }

            return json.dumps(info_dict, indent=2)

        except Exception as e:
            return f"Error loading dataset: {str(e)}"

class DataFrameAnalysisTool:
    """Tool for analyzing pandas DataFrames."""

    def run(self, query: str) -> str:
        """
        Analyze a DataFrame using the specified query.

        The query should be in the format:
        {
            "operation": "operation_name",
            "parameters": {"param1": "value1", ...}
        }

        Supported operations:
        - "load_temp": Load the temporary DataFrame
        - "describe": Statistical description of the DataFrame
        - "correlation": Correlation matrix of numerical columns
        - "groupby": Group data by specified columns
        - "query": Filter DataFrame using pandas query syntax
        - "custom": Execute a custom analysis (specify in detail)

        Args:
            query (str): JSON string with operation and parameters

        Returns:
            str: Analysis results in JSON format
        """
        try:
            # Parse the query JSON
            query_dict = json.loads(query)
            operation = query_dict.get("operation")
            params = query_dict.get("parameters", {})

            # Load the DataFrame from the temporary file
            temp_path = params.get("temp_path", "temp_dataframe.csv")
            if not os.path.exists(temp_path):
                return f"Error: Temporary DataFrame file not found at {temp_path}"

            df = pd.read_csv(temp_path)

            # Execute the requested operation
            if operation == "describe":
                result = df.describe(percentiles=params.get("percentiles", [0.25, 0.5, 0.75])).to_dict()
                return json.dumps(result, indent=2)

            elif operation == "correlation":
                # Select only numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                if numeric_df.empty:
                    return "No numeric columns found for correlation analysis"

                corr_matrix = numeric_df.corr().to_dict()
                return json.dumps(corr_matrix, indent=2)

            elif operation == "groupby":
                if "columns" not in params:
                    return "Error: 'columns' parameter required for groupby operation"

                groupby_cols = params["columns"]
                agg_func = params.get("aggregation", "mean")

                if isinstance(groupby_cols, str):
                    groupby_cols = [groupby_cols]

                result = df.groupby(groupby_cols).agg(agg_func).reset_index().to_dict(orient='records')
                return json.dumps(result, indent=2)

            elif operation == "query":
                if "filter" not in params:
                    return "Error: 'filter' parameter required for query operation"

                filtered_df = df.query(params["filter"])
                result = {
                    "shape": filtered_df.shape,
                    "data": filtered_df.head(params.get("limit", 10)).to_dict(orient='records')
                }
                return json.dumps(result, indent=2)

            elif operation == "summary":
                # Provide a comprehensive summary of the dataset
                summary = {
                    "shape": df.shape,
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "missing_values": df.isnull().sum().to_dict(),
                    "numeric_summary": df.describe().to_dict(),
                    "categorical_summary": {
                        col: df[col].value_counts().to_dict()
                        for col in df.select_dtypes(include=['object', 'category']).columns
                    }
                }
                return json.dumps(summary, indent=2)

            elif operation == "custom":
                # For custom operations, we'll need detailed instructions
                if "code" not in params:
                    return "Error: 'code' parameter required for custom operation"

                # WARNING: Executing arbitrary code can be dangerous
                # In a production environment, you'd want to sanitize and validate this
                result = eval(params["code"])

                if isinstance(result, pd.DataFrame):
                    result = result.to_dict(orient='records')

                return json.dumps(result, indent=2)

            else:
                return f"Error: Unsupported operation '{operation}'"

        except Exception as e:
            return f"Error during DataFrame analysis: {str(e)}"

class DataVisualizationTool:
    """Tool for creating visualizations from DataFrames."""

    def run(self, query: str) -> str:
        """
        Create visualizations from DataFrame data.

        The query should be in the format:
        {
            "plot_type": "plot_name",
            "parameters": {"param1": "value1", ...},
            "save_path": "path/to/save.png" (optional)
        }

        Supported plot types:
        - "histogram": Histogram of a column
        - "scatter": Scatter plot of two columns
        - "bar": Bar chart
        - "line": Line chart
        - "correlation_heatmap": Heatmap of correlation matrix
        - "boxplot": Box plot of a column

        Args:
            query (str): JSON string with plot type and parameters

        Returns:
            str: Path to the saved visualization or error message
        """
        try:
            # Parse the query JSON
            query_dict = json.loads(query)
            plot_type = query_dict.get("plot_type")
            params = query_dict.get("parameters", {})
            save_path = query_dict.get("save_path", "visualization.png")

            # Load the DataFrame from the temporary file
            temp_path = params.get("temp_path", "temp_dataframe.csv")
            if not os.path.exists(temp_path):
                return f"Error: Temporary DataFrame file not found at {temp_path}"

            df = pd.read_csv(temp_path)

            # Set up the figure
            plt.figure(figsize=(10, 6))

            # Create the requested plot
            if plot_type == "histogram":
                if "column" not in params:
                    return "Error: 'column' parameter required for histogram"

                sns.histplot(data=df, x=params["column"], kde=params.get("kde", False))
                plt.title(f"Histogram of {params['column']}")
                plt.xlabel(params["column"])
                plt.ylabel("Frequency")

            elif plot_type == "scatter":
                if "x" not in params or "y" not in params:
                    return "Error: 'x' and 'y' parameters required for scatter plot"

                hue = params.get("hue")
                if hue and hue in df.columns:
                    sns.scatterplot(data=df, x=params["x"], y=params["y"], hue=hue)
                else:
                    sns.scatterplot(data=df, x=params["x"], y=params["y"])

                plt.title(f"Scatter Plot: {params['y']} vs {params['x']}")
                plt.xlabel(params["x"])
                plt.ylabel(params["y"])

            elif plot_type == "bar":
                if "x" not in params or "y" not in params:
                    return "Error: 'x' and 'y' parameters required for bar chart"

                sns.barplot(data=df, x=params["x"], y=params["y"])
                plt.title(f"Bar Chart: {params['y']} by {params['x']}")
                plt.xlabel(params["x"])
                plt.ylabel(params["y"])

                # Rotate x-labels if there are many categories
                if df[params["x"]].nunique() > 5:
                    plt.xticks(rotation=45, ha='right')

            elif plot_type == "line":
                if "x" not in params or "y" not in params:
                    return "Error: 'x' and 'y' parameters required for line chart"

                # Ensure data is sorted by x for proper line plot
                plot_df = df.sort_values(by=params["x"])
                plt.plot(plot_df[params["x"]], plot_df[params["y"]])

                plt.title(f"Line Chart: {params['y']} vs {params['x']}")
                plt.xlabel(params["x"])
                plt.ylabel(params["y"])

            elif plot_type == "correlation_heatmap":
                # Select only numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                if numeric_df.empty:
                    return "No numeric columns found for correlation heatmap"

                # Calculate correlation matrix
                corr_matrix = numeric_df.corr()

                # Create heatmap
                sns.heatmap(corr_matrix, annot=params.get("show_values", True),
                           cmap=params.get("colormap", "coolwarm"),
                           linewidths=0.5)

                plt.title("Correlation Matrix Heatmap")

            elif plot_type == "boxplot":
                if "column" not in params:
                    return "Error: 'column' parameter required for boxplot"

                sns.boxplot(data=df, y=params["column"], x=params.get("group_by"))
                plt.title(f"Boxplot of {params['column']}")

            else:
                return f"Error: Unsupported plot type '{plot_type}'"

            # Save the figure
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            # Return the path to the saved visualization
            return f"Visualization saved to {save_path}"

        except Exception as e:
            return f"Error creating visualization: {str(e)}"