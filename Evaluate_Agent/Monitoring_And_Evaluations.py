# Databricks notebook source
# MAGIC %md # Monitoring and  Evaluations from documents
# MAGIC
# MAGIC This notebook shows how you can synthesize evaluations for an agent that uses document retrieval. It uses the `generate_evals_df` method that is part of the `databricks-agents` Python package.

# COMMAND ----------

# DBTITLE 1,Install and Update Required Python Libraries
#%pip install mlflow mlflow[databricks] databricks-agents databricks-sdk
%pip install -U -qqqq databricks-sdk[openai] backoff
%pip install -U -qqqq mlflow langchain langgraph==0.3.4 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] uv


dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Documentation 
# MAGIC
# MAGIC The API is shown below. For more details, see the documentation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/synthesize-evaluation-set.html) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/synthesize-evaluation-set)).  
# MAGIC
# MAGIC API:
# MAGIC ```py
# MAGIC def generate_evals_df(
# MAGIC     docs: Union[pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821
# MAGIC     *,
# MAGIC         num_evals: int,
# MAGIC     agent_description: Optional[str] = None,
# MAGIC     question_guidelines: Optional[str] = None,
# MAGIC ) -> pd.DataFrame:
# MAGIC     """
# MAGIC     Generate an evaluation dataset with questions and expected answers.
# MAGIC     Generated evaluation set can be used with Databricks Agent Evaluation
# MAGIC     AWS: (https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluate-agent.html)
# MAGIC     Azure: (https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/evaluate-agent).
# MAGIC
# MAGIC     :param docs: A pandas/Spark DataFrame with a text column `content` and a `doc_uri` column.
# MAGIC     :param num_evals: The number of questions (and corresponding answers) to generate in total.
# MAGIC     :param agent_description: Optional, a task description of the agent.
# MAGIC     :param question_guidelines: Optional guidelines to help guide the synthetic question generation. This is a free-form string that will
# MAGIC         be used to prompt the generation. The string can be formatted in markdown and may include sections like:
# MAGIC         - User Personas: Types of users the agent should support
# MAGIC         - Example Questions: Sample questions to guide generation
# MAGIC         - Additional Guidelines: Extra rules or requirements
# MAGIC     """
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC The following code block synthesizes evaluations from a DataFrame of documents.  
# MAGIC - The input can be a Pandas DataFrame or a Spark DataFrame.  
# MAGIC - The output DataFrame can be directly used with `mlflow.evaluate()`.

# COMMAND ----------

# DBTITLE 1,Load Global Configuration Settings
# MAGIC %run ../global_config

# COMMAND ----------

# DBTITLE 1,Monitor Configuration for Fluke Schema Agent
# from databricks.agents.evals.monitors import create_monitor, get_monitor, update_monitor, delete_monitor

# # Get the current monitor configuration 
# #TODO Change the endpoint name
# monitor = get_monitor(endpoint_name="agents_ankit_yadav-uct_schema-vs_agent")

# COMMAND ----------

# DBTITLE 1,Add Evaluation Metrics to Monitor Configuration
# # Update the monitor to add evaluation metrics
# monitor = update_monitor(
#     endpoint_name="agents_ankit_yadav-uct_schema-vs_agent",
#     monitoring_config={
#         "sample": 1,  # Sample 100% of requests - this can be any number from 0 (0%) to 1 (100%).
#         # Select 0+ of Agent Evaluation's built-in judges
#         "metrics": ['guideline_adherence', 'groundedness', 'safety', 'relevance_to_query', 'chunk_relevance'],
#         # Customize these guidelines based on your business requirements.  These guidelines will be analyzed using Agent Evaluation's built in guideline_adherence judge
#         "global_guidelines": {
#             "english": ["The response must be in English."],
#             "clarity": ["The response must be clear, coherent, and concise."],
#             "relevant_if_not_refusal": ["Determine if the response provides an answer to the user's request.  A refusal to answer is considered relevant.  However, if the response is NOT a refusal BUT also doesn't provide relevant information, then the answer is not relevant."],
#             "no_answer_if_no_docs": ["If the agent can not find a relevant document, it should refuse to answer the question and not discuss the reasons why it could not answer."]
#         }
#     }
# )

# COMMAND ----------

# DBTITLE 1,Deploy and Query Fluke Warranty Chatbot
# from mlflow import deployments

# client = deployments.get_deploy_client("databricks")

# questions = [
#     "What are the port types and descriptions for the Motorized APC butterfly valve?"
# ]

# for i, question in enumerate(questions, 1):
#     print(f"\nQuestion {i}: {question}")  
#     response = client.predict(
#         endpoint="agents_ankit_yadav-uct_schema-vs_agent",
#         inputs={
#             "messages": [
#                 {"role": "user", "content": question}
#             ]
#         }
#     )
#     print(response)
    

# COMMAND ----------

# DBTITLE 1,Generate and Evaluate RAG Chatbot Performance
import mlflow
from databricks.agents.evals import generate_evals_df
import pandas as pd
#TODO Change UC_CATALOG, UC_SCHEMA and DOCT_DATA_TABLE Names
docs = spark.sql(f"SELECT doc_content as content, path as doc_uri FROM {UC_CATALOG_NAME}.{UC_SCHEMA_NAME}.{DOCS_DATA_TABLE_NAME}")
display(docs)


agent_description = """
The Agent is a RAG chatbot that answers questions about using Ultra Clean tech Products. The Agent has access to a corpus of Ultra tech equipment manuals, and its task is to answer the user's questions by retrieving the relevant docs from the corpus and synthesizing a helpful, accurate response. The corpus covers a lot of info, but the Agent is specifically designed to interact with Support agents who have questions about Ultra Tech equipment from their customers. So questions outside of this scope are considered irrelevant.
"""
question_guidelines = """
# User personas
- A Support engineer who is trying to answer questions asked by customers
- An experienced, highly technical end customer who might have questions around Fluke equipment

# Example questions
- What is the use of the HRG regulators?
- How can I run the Resistance Verification Test on a 5730A?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

num_evals = 10

evals = generate_evals_df(
    docs,
    # The total number of evals to generate. The method attempts to generate evals that have full coverage over the documents
    # provided. If this number is less than the number of documents, some documents will not have any evaluations generated. 
    # For details about how `num_evals` is used to distribute evaluations across the documents, 
    # see the documentation: 
    # AWS: https://docs.databricks.com/en/generative-ai/agent-evaluation/synthesize-evaluation-set.html#num-evals. 
    # Azure: https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/synthesize-evaluation-set 
    num_evals=num_evals,
    # A set of guidelines that help guide the synthetic generation. This is a free-form string that will be used to prompt the generation.
    agent_description=agent_description,
    question_guidelines=question_guidelines
)

display(evals)

# COMMAND ----------

# Evaluate the model using the newly generated evaluation set. After the function call completes, click the UI link to see the results. You can use this as a baseline for your agent.
results = mlflow.evaluate(
  #TODO Change the endpoint name
  model="endpoints:/agents_ankit_yadav-uct_schema-vs_agent",
  data=evals,
  model_type="databricks-agent"
)

display(results.tables['eval_results'])

# Note: To use a different model serving endpoint, use the following snippet to define an agent_fn. Then, specify that function using the `model` argument.
# MODEL_SERVING_ENDPOINT_NAME = '...'
# def agent_fn(input):
#   client = mlflow.deployments.get_deploy_client("databricks")
#   return client.predict(endpoint=MODEL_SERVING_ENDPOINT_NAME, inputs=input)

# COMMAND ----------

# DBTITLE 1,Display Non-Skipped Evaluated Traces from Delta Table
# # Read evaluated traces from Delta
# display(spark.table(monitor.evaluated_traces_table).filter("evaluation_status != 'skipped'"))
