# TDPrepView MCP Server

⚠️ **ALPHA SOFTWARE - DEMO USE ONLY - NOT FOR PRODUCTION**

MCP server providing ML data preprocessing pipeline and model training tools for Teradata databases.

## Features

- Upload datasets (iris, diabetes, wine, breast_cancer, california_housing, titanic, adult_census) to Teradata
- Create ML preprocessing pipelines with automatic feature engineering
- Generate interactive Sankey diagrams for pipeline visualization
- Train Random Forest models (classification/regression)
- Deploy models as database views using ONNX/BYOM
- Make predictions through deployed model endpoints

## Installation

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd tdprepview-mcp
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up environment variables for database connection (see Configuration section below)

## Configuration for Claude Desktop (macOS)

Add the following configuration to your Claude Desktop config file located at:
`~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "tdprepview": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/YOUR_USERNAME/path/to/tdprepview-mcp",
        "run",
        "python",
        "server.py"
      ],
      "env": {
        "DB_HOST": "your-teradata-host.com",
        "DB_USER": "your_username",
        "DB_PASSWORD": "your_password"
      }
    }
  }
}
```

### Important Notes:

1. **Replace the path**: Change `/Users/YOUR_USERNAME/path/to/tdprepview-mcp` to the actual path where you cloned this repository.

2. **Set your database credentials**: Replace the environment variables with your actual Teradata connection details:
   - `DB_HOST`: Your Teradata server hostname or IP
   - `DB_USER`: Your Teradata username
   - `DB_PASSWORD`: Your Teradata password


## Available Tools

- `get_dummy_data_upload` - Upload datasets to Teradata with automatic indexing
- `create_ml_autoprep_pipeline` - Create and fit preprocessing pipelines
- `save_pipeline_sankey_file` - Generate interactive pipeline visualizations  
- `deploy_pipeline_to_database` - Deploy pipelines as database views
- `train_random_forest_model` - Train ML models on preprocessed data
- `deploy_model_to_teradata` - Deploy ONNX models using BYOM
- `make_predictions` - Test model endpoints with sample data

## Example Workflow

```
1. Upload dataset: "Upload the boston housing dataset to my database"
2. Create pipeline: "Create a preprocessing pipeline for this boston housing table"  
3. Generate viz: "Save a Sankey diagram for this pipeline"
4. Deploy pipeline: "Deploy the pipeline as a view "
5. Train model: "Train a classification model on it"
6. Deploy model: "Deploy this model to Teradata"
7. Test predictions: "Make some test predictions using the deployed model"
```

## Example Execution in Claude Desktop:

[Link to Chat Example using this MCP](https://claude.ai/share/37c480c0-487d-4b20-9414-3ab90e872d1b)