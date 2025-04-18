# About supervised fine-tuning for Gemini models

https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning#2-0-flash

To see an example of supervised fine tuning, run the "Supervised Fine Tuning with Gemini 2.0 Flash for Article Summarization" Jupyter notebook in one of the following environments:

Open in Colab | Open in Colab Enterprise | Open in Vertex AI Workbench user-managed notebooks | View on GitHub

Supervised fine-tuning is a good option when you have a well-defined task with available labeled data. It's particularly effective for domain-specific applications where the language or content significantly differs from the data the large model was originally trained on. You can tune text, image, audio, and document data types.

Supervised fine-tuning adapts model behavior with a labeled dataset. This process adjusts the model's weights to minimize the difference between its predictions and the actual labels. For example, it can improve model performance for the following types of tasks:

    Classification
    Summarization
    Extractive question answering
    Chat

For a discussion of the top tuning use cases, check out the blog post Hundreds of organizations are fine-tuning Gemini models. Here's their favorite use cases.

To learn more, see When to use supervised fine-tuning for Gemini.
Supported models

The following Gemini models support supervised tuning:

    Gemini 2.0 Flash-Lite
    Gemini 2.0 Flash

Limitations
Gemini 2.0 Flash-Lite
Gemini 2.0 Flash
Specification 	Value
Maximum input and output training tokens 	131,072
Maximum input and output serving tokens 	131,072
Maximum validation dataset size 	5000 examples
Maximum training dataset file size 	1GB for JSONL
Maximum training dataset size 	1M text-only examples or 300K multimodal examples
Adapter size 	Supported values are 1, 2, 4, and 8.
Known issues

    A tuned Gemini model can't be deleted from Vertex AI Model Registry. However, as long as it's idle, it won't incur any inference costs.
    Applying controlled generation when submitting inference requests to tuned Gemini models can result in decreased model quality due to data misalignment during tuning and inference time. During tuning, controlled generation isn't applied, so the tuned model isn't able to handle controlled generation well at inference time. Supervised fine-tuning effectively customizes the model to generate structured output. Therefore you don't need to apply controlled generation when making inference requests on tuned models.

Use cases for using supervised fine-tuning

Foundation models work well when the expected output or task can be clearly and concisely defined in a prompt and the prompt consistently produces the expected output. If you want a model to learn something niche or specific that deviates from general patterns, then you might want to consider tuning that model. For example, you can use model tuning to teach the model the following:

    Specific structures or formats for generating output.
    Specific behaviors such as when to provide a terse or verbose output.
    Specific customized outputs for specific types of inputs.

The following examples are use cases that are difficult to capture with only prompt instructions:

    Classification: The expected response is a specific word or phrase.

    Prompt:
    Classify the following text into one of the following classes:
    [business, entertainment].
    Text: Diversify your investment portfolio
      

    Response:
    business
      

    Tuning the model can help prevent the model from generating verbose responses.

    Summarization: The summary follows a specific format. For example, you might need to remove personally identifiable information (PII) in a chat summary.

    Prompt:
    Summarize:
    Jessica: That sounds great! See you in Times Square!
    Alexander: See you at 10!
      

    Response:
    #Person1 and #Person2 agree to meet at Times Square at 10:00 AM.
      

    This formatting of replacing the names of the speakers with #Person1 and #Person2 is difficult to describe and the foundation model might not naturally produce such a response.

    Extractive question answering: The question is about a context and the answer is a substring of the context.

    Prompt:
    Context: There is evidence that there have been significant changes in Amazon rainforest vegetation over the last 21,000 years through the Last Glacial Maximum (LGM) and subsequent deglaciation.
    Question: What does LGM stand for?
      

    Response:
    Last Glacial Maximum
      

    The response "Last Glacial Maximum" is a specific phrase from the context.

    Chat: You need to customize model response to follow a persona, role, or character.

    Prompt:
    User: What's the weather like today?
      

    Response:
    Assistant: As the virtual shopkeeper of Example Organization, I can only help you with the purchases and shipping.
      

You can also tune a model in the following situations:

    Prompts are not producing the expected results consistently enough.
    The task is too complicated to define in a prompt. For example, you want the model to do behavior cloning for a behavior that's hard to articulate in a prompt.
    You have complex intuitions about a task that are difficult to formalize in a prompt.
    You want to reduce the context length by removing the few-shot examples.

Configure a tuning job region

User data, such as the transformed dataset and the tuned model, is stored in the tuning job region. During tuning, computation could be offloaded to other US or EU regions for available accelerators. The offloading is transparent to users.

    If you use the Vertex AI SDK, you can specify the region at initialization. For example:

import vertexai
vertexai.init(project='myproject', location='us-central1')

If you create a supervised fine-tuning job by sending a POST request using the tuningJobs.create method, then you use the URL to specify the region where the tuning job runs. For example, in the following URL, you specify a region by replacing both instances of TUNING_JOB_REGION with the region where the job runs.

 https://TUNING_JOB_REGION

-aiplatform.googleapis.com/v1/projects/PROJECT_ID

/locations/TUNING_JOB_REGION

    /tuningJobs

    If you use the Google Cloud console, you can select the region name in the Region drop-down field on the Model details page. This is the same page where you select the base model and a tuned model name.

Quota

Quota is enforced on the number of concurrent tuning jobs. Every project comes with a default quota to run at least one tuning job. This is a global quota, shared across all available regions and supported models. If you want to run more jobs concurrently, you need to request additional quota for Global concurrent tuning jobs.
Pricing

Pricing for tuning Gemini models can be found here: Vertex AI pricing.

Training tokens are calculated by the total number of tokens in your training dataset, multiplied by your number of epochs. For all models, after tuning, inference costs for the tuned model still apply. Inference pricing is the same for each stable version of Gemini. For more information, see Vertex AI pricing and Available Gemini stable model versions



# Text tuning

https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune_gemini/text_tune

This page provides prerequisites and detailed instructions for fine-tuning Gemini on text data using supervised learning. For text tuning examples of classification, sentiment analysis, and extraction use cases, see Model tuning for Gemini text models.
Use cases

Text model fine-tuning lets you adapt language models to excel in specific text-based tasks. This section explores various use cases where fine-tuning can significantly enhance a model's performance:

    Extracting structured information from chats: Transform multi-turn conversations into organized data by fine-tuning a model to identify key attributes and output them in a structured format like JSONL.
    Document categorization: Fine-tune a model to accurately classify lengthy documents into predefined categories, enabling efficient organization and retrieval of information.
    Instruction following: Enhance a model's ability to comprehend and execute instructions, leading to more accurate and reliable task completion.
    Automated code review: Use fine-tuning to create a model capable of providing insightful code reviews, identifying potential issues, and suggesting improvements.
    Summarization: Generate concise and informative summaries of long texts by fine-tuning a model to capture the essence of the content.
    Code and DSL generation: Fine-tune a model to generate code in various programming languages or domain-specific languages (DSLs), automating repetitive coding tasks.
    Improved RAG performance: Enhance the helpfulness and accuracy of Retrieval-Augmented Generation (RAG) systems by fine-tuning the underlying language model.

Dataset format

The fileUri for your dataset can be the URI for a file in a Cloud Storage bucket, or it can be a publicly available HTTP or HTTPS URL.

The following is an example of a text dataset.

To see the generic format example, see Dataset example for Gemini.

{
  "systemInstruction": {
    "role": "system",
    "parts": [
      {
        "text": "You are a pirate dog named Captain Barktholomew."
      }
    ]
  },
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Hi"
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "text": "Argh! What brings ye to my ship?"
        }
      ]
    },
    {
      "role": "user",
      "parts": [
        {
          "text": "What's your name?"
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "text": "I be Captain Barktholomew, the most feared pirate dog of the seven seas."
        }
      ]
    }
  ]
}

Sample datasets

You can use the following sample datasets to learn how to tune a Gemini model.

    Sample tuning dataset
    Sample validation dataset

To use these datasets, specify the URIs in the applicable parameters when creating a text model supervised fine-tuning job.

For example:

...
"training_dataset_uri": "gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl",
...
"validation_dataset_uri": "gs://cloud-samples-data/ai-platform/generative_ai/sft_validation_data.jsonl",
...

Estimate the cost of tuning with a dataset

The following notebook can help you estimate token counts and tuning costs when running a tuning job for Gemini models.

To see an example of Vertex AI Supervised Tuning token count and cost estimation, run the "Vertex AI Supervised Tuning token count and cost estimation." Jupyter notebook in one of the following environments:

Open in Colab | Open in Colab Enterprise | Open in Vertex AI Workbench user-managed notebooks | View on GitHub
What's next

    To start tuning, see Tune Gemini models by using supervised fine-tuning.
    To learn how supervised fine-tuning can be used in a solution that builds a generative AI knowledge base, see Jump Start Solution: Generative AI knowledge base.



Supervised Fine Tuning with Gemini 2.0 Flash for Article Summarization
Google Colaboratory logo
Open in Colab 	Google Cloud Colab Enterprise logo
Open in Colab Enterprise 	Vertex AI logo
Open in Workbench 	GitHub logo
View on GitHub

Share to:
LinkedIn logo Bluesky logo X logo Reddit logo Facebook logo
Author(s)
Deepak Moonat
Safiuddin Khaja
Overview

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the various Gemini models, such as Gemini 2.0 Pro/Flash, Gemini 2.0/Flash, Gemini/Flash and more.

This notebook demonstrates how to fine-tune the Gemini 2.0 Flash generative model using the Vertex AI Supervised Tuning feature. Supervised Tuning allows you to use your own training data to further refine the base model's capabilities towards your specific tasks.

Supervised Tuning uses labeled examples to tune a model. Each example demonstrates the output you want from your text model during inference.

First, ensure your training data is of high quality, well-labeled, and directly relevant to the target task. This is crucial as low-quality data can adversely affect the performance and introduce bias in the fine-tuned model.

    Training: Experiment with different configurations to optimize the model's performance on the target task.
    Evaluation:
        Metric: Choose appropriate evaluation metrics that accurately reflect the success of the fine-tuned model for your specific task
        Evaluation Set: Use a separate set of data to evaluate the model's performance

Refer to public documentation for more details.

Before running this notebook, ensure you have:

    A Google Cloud project: Provide your project ID in the PROJECT_ID variable.

    Authenticated your Colab environment: Run the authentication code block at the beginning.

    Prepared training data (Test with your own data or use the one in the notebook): Data should be formatted in JSONL with prompts and corresponding completions.

Objective

In this tutorial, you will learn how to use Vertex AI to tune a Gemini 2.0 Flash model.

This tutorial uses the following Google Cloud ML services:

    Vertex AI

The steps performed include:

    Prepare and load the dataset
    Load the gemini-2.0-flash-001 model
    Evaluate the model before tuning
    Tune the model.
        This will automatically create a Vertex AI endpoint and deploy the model to it
    Make a prediction using tuned model
    Evaluate the model after tuning

Costs

This tutorial uses billable components of Google Cloud:

    Vertex AI
    Cloud Storage

Learn about Vertex AI pricing, Cloud Storage pricing, and use the Pricing Calculator to generate a cost estimate based on your projected usage.
Wikilingua Dataset

The dataset includes article and summary pairs from WikiHow. It consists of article-summary pairs in multiple languages. Refer to the following github repository for more details.

For this notebook, we have picked english language dataset.
Dataset Citation

@inproceedings{ladhak-wiki-2020,
    title={WikiLingua: A New Benchmark Dataset for Multilingual Abstractive Summarization},
    author={Faisal Ladhak, Esin Durmus, Claire Cardie and Kathleen McKeown},
    booktitle={Findings of EMNLP, 2020},
    year={2020}
}

Getting Started
Install Gen AI SDK and other required packages

The new Google Gen AI SDK provides a unified interface to Gemini through both the Gemini Developer API and the Gemini API on Vertex AI. With a few exceptions, code that runs on one platform will run on both. This means that you can prototype an application using the Developer API and then migrate the application to Vertex AI without rewriting your code.

%pip install --upgrade --user --quiet google-genai google-cloud-aiplatform rouge_score plotly jsonlines

Restart runtime (Colab only)

To use the newly installed packages, you must restart the runtime on Google Colab.

import sys

if "google.colab" in sys.modules:
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)

⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️
Step0: Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the cell below to authenticate your environment.

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()


    If you are running this notebook in a local development environment:

        Install the Google Cloud SDK.

        Obtain authentication credentials. Create local credentials by running the following command and following the oauth2 flow (read more about the command here):

        gcloud auth application-default login

Step1: Import Libraries

import time

from google import genai

# For extracting vertex experiment details.
from google.cloud import aiplatform
from google.cloud.aiplatform.metadata import context
from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.genai import types

# For data handling.
import jsonlines
import pandas as pd

# For visualization.
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For evaluation metric computation.
from rouge_score import rouge_scorer
from tqdm import tqdm

# For fine tuning Gemini model.
import vertexai

Step2: Set Google Cloud project information and initialize Vertex AI and Gen AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and enable the Vertex AI API.

Learn more about setting up a project and a development environment.

PROJECT_ID = "[YOUR_PROJECT_ID]"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}


vertexai.init(project=PROJECT_ID, location=REGION)

client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

Step3: Create Dataset in correct format

The dataset used to tune a foundation model needs to include examples that align with the task that you want the model to perform. Structure your training dataset in a text-to-text format. Each record, or row, in the dataset contains the input text (also referred to as the prompt) which is paired with its expected output from the model. Supervised tuning uses the dataset to teach the model to mimic a behavior, or task, you need by giving it hundreds of examples that illustrate that behavior.

Your dataset size depends on the task, and follows the recommendation mentioned in the Overview section. The more examples you provide in your dataset, the better the results.
Dataset format

Training data should be structured within a JSONL file located at a Google Cloud Storage (GCS) URI. Each line (or row) of the JSONL file must adhere to a specific schema: It should contain a contents array, with objects inside defining a role (either "user" for user input or "model" for model output) and parts, containing the input data. For example, a valid data row would look like this:

{
   "contents":[
      {
         "role":"user",  # This indicate input content
         "parts":[
            {
               "text":"How are you?"
            }
         ]
      },
      {
         "role":"model", # This indicate target content
         "parts":[ # text only
            {
               "text":"I am good, thank you!"
            }
         ]
      }
      #  ... repeat "user", "model" for multi turns.
   ]
}

Refer to the public documentation for more details.

To run a tuning job, you need to upload one or more datasets to a Cloud Storage bucket. You can either create a new Cloud Storage bucket or use an existing one to store dataset files. The region of the bucket doesn't matter, but we recommend that you use a bucket that's in the same Google Cloud project where you plan to tune your model.
Step3 [a]: Create a Cloud Storage bucket

Create a storage bucket to store intermediate artifacts such as datasets.

# Provide a bucket name
BUCKET_NAME = "[YOUR_BUCKET_NAME]"  # @param {type:"string"}
BUCKET_URI = f"gs://{BUCKET_NAME}"


Only if your bucket doesn't already exist: Run the following cell to create your Cloud Storage bucket.

! gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}

Step3 [b]: Upload tuning data to Cloud Storage

    Data used in this notebook is present in the public Google Cloud Storage(GCS) bucket.
    It's in Gemini finetuning dataset format


!gsutil ls gs://github-repo/generative-ai/gemini/tuning/summarization/wikilingua


!gsutil cp gs://github-repo/generative-ai/gemini/tuning/summarization/wikilingua/* .

Convert Gemini tuning dataset to Gemini 2.0 tuning dataset format

def save_jsonlines(file, instances):
    """
    Saves a list of json instances to a jsonlines file.
    """
    with jsonlines.open(file, mode="w") as writer:
        writer.write_all(instances)


def create_tuning_samples(file_path):
    """
    Creates tuning samples from a file.
    """
    with jsonlines.open(file_path) as reader:
        instances = []
        for obj in reader:
            instance = []
            for content in obj["messages"]:
                instance.append(
                    {"role": content["role"], "parts": [{"text": content["content"]}]}
                )
            instances.append({"contents": instance})
    return instances


train_file = "sft_train_samples.jsonl"
train_instances = create_tuning_samples(train_file)
len(train_instances)


# save the training instances to jsonl file
save_jsonlines(train_file, train_instances)


val_file = "sft_val_samples.jsonl"
val_instances = create_tuning_samples(val_file)
len(val_instances)


# save the validation instances to jsonl file
save_jsonlines(val_file, val_instances)


# Copy the tuning and evaluation data to your bucket.
!gsutil cp {train_file} {BUCKET_URI}/train/
!gsutil cp {val_file} {BUCKET_URI}/val/

Step3 [c]: Test dataset

    It contains document text(input_text) and corresponding reference summary(output_text), which will be compared with the model generated summary


# Load the test dataset using pandas as it's in the csv format.
testing_data_path = "sft_test_samples.csv"
test_data = pd.read_csv(testing_data_path)
test_data.head()


test_data.loc[0, "input_text"]


# Article summary stats
stats = test_data["output_text"].apply(len).describe()
stats


print(f"Total `{stats['count']}` test records")
print(f"Average length is `{stats['mean']}` and max is `{stats['max']}` characters")
print("\nConsidering 1 token = 4 chars")

# Get ceil value of the tokens required.
tokens = (stats["max"] / 4).__ceil__()
print(
    f"\nSet max_token_length = stats['max']/4 = {stats['max']/4} ~ {tokens} characters"
)
print(f"\nLet's keep output tokens upto `{tokens}`")


# Maximum number of tokens that can be generated in the response by the LLM.
# Experiment with this number to get optimal output.
max_output_tokens = tokens

Step4: Initailize model

The following Gemini text model support supervised tuning:

    gemini-2.0-flash-001


base_model = "gemini-2.0-flash-001"

Step5: Test the Gemini model
Generation config

    Each call that you send to a model includes parameter values that control how the model generates a response. The model can generate different results for different parameter values
    Experiment with different parameter values to get the best values for the task

Refer to the following link for understanding different parameters

Prompt is a natural language request submitted to a language model to receive a response back

Some best practices include

    Clearly communicate what content or information is most important
    Structure the prompt:
        Defining the role if using one. For example, You are an experienced UX designer at a top tech company
        Include context and input data
        Provide the instructions to the model
        Add example(s) if you are using them

Refer to the following link for prompt design strategies.

Wikilingua data contains the following task prompt at the end of the article, Provide a summary of the article in two or three sentences:

test_doc = test_data.loc[0, "input_text"]

prompt = f"""
{test_doc}
"""

config = {
    "temperature": 0.1,
    "max_output_tokens": max_output_tokens,
}

response = client.models.generate_content(
    model=base_model,
    contents=prompt,
    config=config,
).text
print(response)


# Ground truth
test_data.loc[0, "output_text"]

Step6: Evaluation before model tuning

    Evaluate the Gemini model on the test dataset before tuning it on the training dataset.


# Convert the pandas dataframe to records (list of dictionaries).
corpus = test_data.to_dict(orient="records")
# Check number of records.
len(corpus)

Evaluation metric

The type of metrics used for evaluation depends on the task that you are evaluating. The following table shows the supported tasks and the metrics used to evaluate each task:
Task 	Metric(s)
Classification 	Micro-F1, Macro-F1, Per class F1
Summarization 	ROUGE-L
Question Answering 	Exact Match
Text Generation 	BLEU, ROUGE-L


Refer to this documentation for metric based evaluation.

    Recall-Oriented Understudy for Gisting Evaluation (ROUGE): A metric used to evaluate the quality of automatic summaries of text. It works by comparing a generated summary to a set of reference summaries created by humans.

Now you can take the candidate and reference to evaluate the performance. In this case, ROUGE will give you:

    rouge-1, which measures unigram overlap
    rouge-2, which measures bigram overlap
    rouge-l, which measures the longest common subsequence

Recall vs. Precision

Recall, meaning it prioritizes how much of the information in the reference summaries is captured in the generated summary.

Precision, which measures how much of the generated summary is relevant to the original text.

Alternate Evaluation method: Check out the AutoSxS evaluation for automatic evaluation of the task.

# Create rouge_scorer object for evaluation
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def run_evaluation(model, corpus: list[dict]) -> pd.DataFrame:
    """Runs evaluation for the given model and data.

    Args:
      model: The generation model.
      corpus: The test data.

    Returns:
      A pandas DataFrame containing the evaluation results.
    """
    records = []
    for item in tqdm(corpus):
        document = item.get("input_text")
        summary = item.get("output_text")

        # Catch any exception that occur during model evaluation.
        try:
            response = client.models.generate_content(
                model=model,
                contents=document,
                config=config,
            )

            # Check if response is generated by the model, if response is empty then continue to next item.
            if not (
                response
                and response.candidates
                and response.candidates[0].content.parts
            ):
                print(
                    f"\nModel has blocked the response for the document.\n Response: {response}\n Document: {document}"
                )
                continue

            # Calculates the ROUGE score for a given reference and generated summary.
            scores = scorer.score(target=summary, prediction=response.text)

            # Append the results to the records list
            records.append(
                {
                    "document": document,
                    "summary": summary,
                    "generated_summary": response.text,
                    "scores": scores,
                    "rouge1_precision": scores.get("rouge1").precision,
                    "rouge1_recall": scores.get("rouge1").recall,
                    "rouge1_fmeasure": scores.get("rouge1").fmeasure,
                    "rouge2_precision": scores.get("rouge2").precision,
                    "rouge2_recall": scores.get("rouge2").recall,
                    "rouge2_fmeasure": scores.get("rouge2").fmeasure,
                    "rougeL_precision": scores.get("rougeL").precision,
                    "rougeL_recall": scores.get("rougeL").recall,
                    "rougeL_fmeasure": scores.get("rougeL").fmeasure,
                }
            )
        except AttributeError as attr_err:
            print("Attribute Error:", attr_err)
            continue
        except Exception as err:
            print("Error:", err)
            continue
    return pd.DataFrame(records)


# Batch of test data.
corpus_batch = corpus[:100]

⚠️ It will take ~2 mins for the evaluation run on the provided batch. ⚠️

# Run evaluation using loaded model and test data corpus
evaluation_df = run_evaluation(base_model, corpus_batch)


evaluation_df.head()


evaluation_df_stats = evaluation_df.dropna().describe()


# Statistics of the evaluation dataframe.
evaluation_df_stats


print("Mean rougeL_precision is", evaluation_df_stats.rougeL_precision["mean"])

Step7: Fine-tune the Model

    source_model: Specifies the base Gemini model version you want to fine-tune.
    train_dataset: Path to your training data in JSONL format.

Optional parameters

    validation_dataset: If provided, this data is used to evaluate the model during tuning.
    tuned_model_display_name: Display name for the tuned model.
    epochs: The number of training epochs to run.
    learning_rate_multiplier: A value to scale the learning rate during training.
    adapter_size : Gemini 2.0 Pro supports Adapter length [1, 2, 4, 8], default value is 4.

Note: The default hyperparameter settings are optimized for optimal performance based on rigorous testing and are recommended for initial use. Users may customize these parameters to address specific performance requirements.

tuned_model_display_name = "[DISPLAY NAME FOR TUNED MODEL]"  # @param {type:"string"}

training_dataset = {
    "gcs_uri": f"{BUCKET_URI}/train/sft_train_samples.jsonl",
}

validation_dataset = types.TuningValidationDataset(
    gcs_uri=f"{BUCKET_URI}/val/sft_val_samples.jsonl"
)

# Tune a model using `tune` method.
sft_tuning_job = client.tunings.tune(
    base_model=base_model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        tuned_model_display_name=tuned_model_display_name,
        validation_dataset=validation_dataset,
    ),
)


# Get the tuning job info.
tuning_job = client.tunings.get(name=sft_tuning_job.name)
tuning_job


Note: Tuning time depends on several factors, such as training data size, number of epochs, learning rate multiplier, etc.
⚠️ It will take ~15 mins for the model tuning job to complete on the provided dataset and set configurations/hyperparameters. ⚠️

%%time
# Wait for job completion

running_states = [
    "JOB_STATE_PENDING",
    "JOB_STATE_RUNNING",
]

while tuning_job.state.name in running_states:
    print(".", end="")
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(10)
print()


tuned_model = tuning_job.tuned_model.endpoint
experiment_name = tuning_job.experiment

print("Tuned model experiment", experiment_name)
print("Tuned model endpoint resource name:", tuned_model)

Step7 [a]: Tuning and evaluation metrics
Model tuning metrics

    /train_total_loss: Loss for the tuning dataset at a training step.
    /train_fraction_of_correct_next_step_preds: The token accuracy at a training step. A single prediction consists of a sequence of tokens. This metric measures the accuracy of the predicted tokens when compared to the ground truth in the tuning dataset.
    /train_num_predictions: Number of predicted tokens at a training step

Model evaluation metrics:

    /eval_total_loss: Loss for the evaluation dataset at an evaluation step.
    /eval_fraction_of_correct_next_step_preds: The token accuracy at an evaluation step. A single prediction consists of a sequence of tokens. This metric measures the accuracy of the predicted tokens when compared to the ground truth in the evaluation dataset.
    /eval_num_predictions: Number of predicted tokens at an evaluation step.

The metrics visualizations are available after the model tuning job completes. If you don't specify a validation dataset when you create the tuning job, only the visualizations for the tuning metrics are available.

# Locate Vertex AI Experiment and Vertex AI Experiment Run
experiment = aiplatform.Experiment(experiment_name=experiment_name)
filter_str = metadata_utils._make_filter_string(
    schema_title="system.ExperimentRun",
    parent_contexts=[experiment.resource_name],
)
experiment_run = context.Context.list(filter_str)[0]


# Read data from Tensorboard
tensorboard_run_name = f"{experiment.get_backing_tensorboard_resource().resource_name}/experiments/{experiment.name}/runs/{experiment_run.name.replace(experiment.name, '')[1:]}"
tensorboard_run = aiplatform.TensorboardRun(tensorboard_run_name)
metrics = tensorboard_run.read_time_series_data()


def get_metrics(metric: str = "/train_total_loss"):
    """
    Get metrics from Tensorboard.

    Args:
      metric: metric name, eg. /train_total_loss or /eval_total_loss.
    Returns:
      steps: list of steps.
      steps_loss: list of loss values.
    """
    loss_values = metrics[metric].values
    steps_loss = []
    steps = []
    for loss in loss_values:
        steps_loss.append(loss.scalar.value)
        steps.append(loss.step)
    return steps, steps_loss


# Get Train and Eval Loss
train_loss = get_metrics(metric="/train_total_loss")
eval_loss = get_metrics(metric="/eval_total_loss")

Step7 [b]: Plot the metrics

# Plot the train and eval loss metrics using Plotly python library

fig = make_subplots(
    rows=1, cols=2, shared_xaxes=True, subplot_titles=("Train Loss", "Eval Loss")
)

# Add traces
fig.add_trace(
    go.Scatter(x=train_loss[0], y=train_loss[1], name="Train Loss", mode="lines"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=eval_loss[0], y=eval_loss[1], name="Eval Loss", mode="lines"),
    row=1,
    col=2,
)

# Add figure title
fig.update_layout(title="Train and Eval Loss", xaxis_title="Steps", yaxis_title="Loss")

# Set x-axis title
fig.update_xaxes(title_text="Steps")

# Set y-axes titles
fig.update_yaxes(title_text="Loss")

# Show plot
fig.show()

Step8: Load the Tuned Model

    Load the fine-tuned model using GenerativeModel class with the tuning job model endpoint name.

    Test the tuned model with the following prompt


prompt


if True:
    # Test with the loaded model.
    print("***Testing***")
    print(
        client.models.generate_content(
            model=tuned_model, contents=prompt, config=config
        ).text
    )
else:
    print("State:", tuning_job.state.name.state)
    print("Error:", tuning_job.state.name.error)


    We can clearly see the difference between summary generated pre and post tuning, as tuned summary is more inline with the ground truth format (Note: Pre and Post outputs, might vary based on the set parameters.)
        Pre: This article describes a method for applying lotion to your back using your forearms as applicators. By squeezing lotion onto your forearms and then reaching behind your back, you can use a windshield wiper motion to spread the lotion across your back. The method acknowledges potential limitations for those with shoulder pain or limited flexibility.
        Post: Squeeze a line of lotion on your forearm. Reach behind you and rub your back.
        Ground Truth: Squeeze a line of lotion onto the tops of both forearms and the backs of your hands. Place your arms behind your back. Move your arms in a windshield wiper motion.

Step9: Evaluation post model tuning
⚠️ It will take ~5 mins for the evaluation on the provided batch. ⚠️

# run evaluation
evaluation_df_post_tuning = run_evaluation(tuned_model, corpus_batch)


evaluation_df_post_tuning.head()


evaluation_df_post_tuning_stats = evaluation_df_post_tuning.dropna().describe()


# Statistics of the evaluation dataframe post model tuning.
evaluation_df_post_tuning_stats


print(
    "Mean rougeL_precision is", evaluation_df_post_tuning_stats.rougeL_precision["mean"]
)

Improvement

improvement = round(
    (
        (
            evaluation_df_post_tuning_stats.rougeL_precision["mean"]
            - evaluation_df_stats.rougeL_precision["mean"]
        )
        / evaluation_df_stats.rougeL_precision["mean"]
    )
    * 100,
    2,
)
print(
    f"Model tuning has improved the rougeL_precision by {improvement}% (result might differ based on each tuning iteration)"
)

Conclusion

Performance could be further improved:

    By adding more training samples. In general, improve your training data quality and/or quantity towards getting a more diverse and comprehensive dataset for your task
    By tuning the hyperparameters, such as epochs and learning rate multiplier
        To find the optimal number of epochs for your dataset, we recommend experimenting with different values. While increasing epochs can lead to better performance, it's important to be mindful of overfitting, especially with smaller datasets. If you see signs of overfitting, reducing the number of epochs can help mitigate the issue
    You may try different prompt structures/formats and opt for the one with better performance

Cleaning up

To clean up all Google Cloud resources used in this project, you can delete the Google Cloud project you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial.

Refer to this instructions to delete the resources from console.

# Delete Experiment.
delete_experiments = True
if delete_experiments:
    experiments_list = aiplatform.Experiment.list()
    for experiment in experiments_list:
        if experiment.resource_name == experiment_name:
            print(experiment.resource_name)
            experiment.delete()
            break

print("***" * 10)

# Delete Endpoint.
delete_endpoint = True
# If force is set to True, all deployed models on this
# Endpoint will be first undeployed.
if delete_endpoint:
    for endpoint in aiplatform.Endpoint.list():
        if endpoint.resource_name == tuned_model:
            print(endpoint.resource_name)
            endpoint.delete(force=True)
            break

print("***" * 10)

# Delete Cloud Storage Bucket.
delete_bucket = True
if delete_bucket:
    ! gsutil -m rm -r $BUCKET_URI



Sample (1 row) of training data:

{"contents": [{"role": "user", "parts": [{"text": "Honesty is usually the best policy. It is disrespectful to lie to someone. If you don't want to date someone, you should say so.  Sometimes it is easy to be honest. For example, you might be able to truthfully say, \"No, thank you, I already have a date for that party.\" Other times, you might need to find a kinder way to be nice. Maybe you are not attracted to the person. Instead of bluntly saying that, try saying, \"No, thank you, I just don't think we would be a good fit.\" Avoid making up a phony excuse. For instance, don't tell someone you will be out of town this weekend if you won't be. There's a chance that you might then run into them at the movies, which would definitely cause hurt feelings. A compliment sandwich is a really effective way to provide feedback. Essentially, you \"sandwich\" your negative comment between two positive things. Try using this method when you need to reject someone.  An example of a compliment sandwich is to say something such as, \"You're an awesome person. Unfortunately, I'm not interested in dating you. Someone else is going to be really lucky to date someone with such a great personality!\" You could also try, \"You are a really nice person. I'm only interested you as a friend. I like when we hang out in big groups together!\" Be sincere. If you offer false compliments, the other person will likely be able to tell and feel hurt. If you do not want to date someone, it is best to be upfront about your feelings. Do not beat around the bush. If your mind is made up, it is best to clearly state your response.  If someone asks you to date them and you don't want to, you can be direct and kind at the same time. State your answer clearly. You can make your feelings clear without purposefully hurting someone else's feelings. Try smiling and saying, \"That sounds fun, but no thank you. I'm not interested in dating you.\" Don't beat around the bush. If you do not want to accept the date, there is no need to say, \"Let me think about it.\" It is best to get the rejection over with. You don't want to give someone false hope. Avoid saying something like, \"Let me check my schedule and get back to you.\" Try to treat the person the way you would want to be treated. This means that you should choose your words carefully. Be thoughtful in your response.  It's okay to pause before responding. You might be taken by surprise and need a moment to collect your thoughts. Say thank you. It is a compliment to be asked out. You can say, \"I'm flattered. Unfortunately, I can't accept.\" Don't laugh. Many people laugh nervously in awkward situations. Try to avoid giggling, as that is likely to result in hurt feelings. Sometimes it is not what you say, but how you say it. If you need to reject someone, think about factors other than your words. Non-verbal communication matters, too.  Use the right tone of voice. Try to sound gentle but firm. Make eye contact. This helps convey that you are being serious, and also shows respect for the other person. If you are in public, try not to speak too loudly. It is not necessary for everyone around you to know that you are turning down a date.\n\nProvide a summary of the article in two or three sentences:\n\n"}]}, {"role": "model", "parts": [{"text": "Tell the truth. Use a \"compliment sandwich\". Be direct. Treat the person with respect. Communicate effectively."}]}]}

Sample (1 row) of validation data:

{"contents": [{"role": "user", "parts": [{"text": "Each step consists of 2 main parts: a riser, which is the vertical part of the step, and the tread, which is the part you step on. Other important terms to know when building your stairs include:  The pitch line, an imaginary line that runs from the lip of the bottom stair to the lip of the topmost step. The “pitch” of the stairs is the angle between the base of the stairs and the pitch line. The rise, which is the total height of your staircase from its base to the topmost step. The run of your stairs, which is the depth of your stair measured from the front of your staircase to its back. The stringers of your stairs are the side supports to either side of each step. The measurements for your stringers will be used in the creation of your concrete forms. First, you'll need to find the total rise of your stairs, which can be found by measuring the total height from the planned base to the higher level/floor you are building up to. Then:  Measure the run of the steps by finding the distance between the frontmost and backmost part of the stair. Find the width of your stairs by measuring the distance between the leftmost and rightmost bounds of your planned steps. You can do this by finding the rise (total height from the bottom level to the top level) of your stairs and dividing that number by the riser height (the height of each step). Standard risers are between 6 to 8 in (15 to 20 cm). The building code in some states sets the maximum riser height at 8.25\" (21 cm), but you should be sure to check your local building code to prevent building stairs that violate state code or local ordinances.  If this number comes out as a decimal, you must round up to the next whole number. If your stairs are built so the topmost tread is a step below the level/floor you are building up to, the number of treads you need will be 1 less than the number of risers. Otherwise, you will need an equal number of treads and risers. There are many online sites that offer simple step calculators to simplify this process. You should be able to find these through a general Internet search for a \"stair calculator.\" These calculators will help you determine the exact specifications of your steps, including the number of treads, height of each riser, stringer length, pitch line, and so on. Divide the total rise (height) of the entire staircase by the number of risers. Generally, you should round the riser height of your steps to the nearest 1/16 inch (1.6 mm). You may have to install brackets in the concrete when you pour, so determine if handrails are a necessity for your stairs. If the stairs are next to a wall, you can always attach a handrail to that.\n\nProvide a summary of the article in two or three sentences:\n\n"}]}, {"role": "model", "parts": [{"text": "Equip yourself with a stair-building vocabulary. Measure the dimensions for your stairs. Calculate the number of stairs you will need. Find the height of your risers. Plan for handrails if you need them."}]}]}


# Supervised fine-tuning with Gemini 2.0 Flash for Q&A using the Google Gen AI SDK

https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/tuning/sft_gemini_qa.ipynb

Gemini is a family of generative AI models developed by Google DeepMind designed for multimodal use cases. The Gemini API gives you access to the various Gemini models, such as Gemini 2.0 and Gemini 2.0. This notebook demonstrates fine-tuning the Gemini 2.0 using the Vertex AI Supervised Tuning feature. Supervised Tuning allows you to use your own labeled training data to further refine the base model's capabilities toward your specific tasks. Supervised Tuning uses labeled examples to tune a model. Each example demonstrates the output you want from your text model during inference. First, ensure your training data is of high quality, well-labeled, and directly relevant to the target task. This is crucial as low-quality data can adversely affect the performance and introduce bias in the fine-tuned model. Training: Experiment with different configurations to optimize the model's performance on the target task. Evaluation: Metric: Choose appropriate evaluation metrics that accurately reflect the success of the fine-tuned model for your specific task Evaluation Set: Use a separate set of data to evaluate the model's performance

Refer to public documentation for more details.

Before running this notebook, ensure you have:

    A Google Cloud project: Provide your project ID in the PROJECT_ID variable.

    Authenticated your Colab environment: Run the authentication code block at the beginning.

    Prepared training data (Test with your own data or use the one in the notebook): Data should be formatted in JSONL with prompts and corresponding completions.

Costs

This tutorial uses billable components of Google Cloud:

    Vertex AI
    Cloud Storage

Learn about Vertex AI pricing, Cloud Storage pricing, and use the Pricing Calculator to generate a cost estimate based on your projected usage.

To estimate the cost of token please have a look at this notebook
Get started
Install the Google Gen AI SDK and other required packages

The new Google Gen AI SDK provides a unified interface to Gemini through both the Gemini Developer API and the Gemini API on Vertex AI. With a few exceptions, code that runs on one platform will run on both. This means that you can prototype an application using the Developer API and then migrate the application to Vertex AI without rewriting your code.

%pip install --upgrade --quiet google-cloud-aiplatform google-genai
     
Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
     
⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️
Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
     
Set the Google Cloud project information and initialize the Google Gen AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and enable the Vertex AI API.

Learn more about setting up a project and a development environment.

# Use the environment variable if the user doesn't provide Project ID.
import os

from google import genai
from google.genai import types

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
     
Import libraries

from collections import Counter
import json
import random

# Vertex AI SDK
from google.cloud import aiplatform
from google.cloud.aiplatform.metadata import context
from google.cloud.aiplatform.metadata import utils as metadata_utils
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
     
Data
SQuAD dataset

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

You can find more information on the SQuAD github page

@inproceedings{rajpurkar-etal-2016-squad,
    title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text",
    author = "Rajpurkar, Pranav  and
      Zhang, Jian  and
      Lopyrev, Konstantin  and
      Liang, Percy",
    editor = "Su, Jian  and
      Duh, Kevin  and
      Carreras, Xavier",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1264",
    doi = "10.18653/v1/D16-1264",
    pages = "2383--2392",
    eprint={1606.05250},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
}

First update the BUCKET_NAME parameter below. You can either use an existing bucket or create a new one.

# Provide a bucket name
BUCKET_NAME = "[your-bucket-name]"  # @param {type:"string"}
BUCKET_URI = f"gs://{BUCKET_NAME}"
print(BUCKET_URI)
     

Only run the code below if you want to create a new Google Cloud Storage bucket.

# ! gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}
     

Next you will copy the data into your bucket.

!gsutil cp gs://github-repo/generative-ai/gemini/tuning/qa/squad_test.csv .
!gsutil cp gs://github-repo/generative-ai/gemini/tuning/qa/squad_train.csv .
!gsutil cp gs://github-repo/generative-ai/gemini/tuning/qa/squad_validation.csv .
     
Baseline

Next you will prepare some data that you will use to establish a baseline. This means evaluating the out of the box default model on a representative sample of your dataset before any fine-tuning. A baseline allows you to quantify the improvements achieved through fine-tuning.

test_df = pd.read_csv("squad_test.csv")
test_df.head(2)
     

First you need to prepare some data to evaluate the out of the box model and set a baseline. In this case, we will lower the text and remove extra whitespace, but preserve newlines.

row_dataset = random.randint(0, 100)  # lets take a random example from the dataset


def normalize_answer(s):
    """Lower text and remove extra whitespace, but preserve newlines."""

    def white_space_fix(text):
        return " ".join(text.split())  # Splits by any whitespace, including \n

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


test_df["answers"] = test_df["answers"].apply(normalize_answer)
     

You want to make sure that you test data looks the same as your training data to prevent training / serving skew. We will add a system instruction to the dataset:

    SystemInstruct: System instructions are a set of instructions that the model processes before it processes prompts. We recommend that you use system instructions to tell the model how you want it to behave and respond to prompts.
    You will also combine the context and question. Both will be send to the model to generate a response.


few_shot_examples = test_df.sample(3)
# Get the indices of the sampled rows
dropped_indices = few_shot_examples.index
# Remove the sampled rows from the original DataFrame
test_df = test_df.drop(dropped_indices)

few_shot_prompt = ""
for _, row in few_shot_examples.iterrows():
    few_shot_prompt += (
        f"Context: {row.context}\nQuestion: {row.question}\nAnswer: {row.answers}\n\n"
    )

print(few_shot_prompt)
     

# Incorporate few-shot examples into the system instruction
systemInstruct = f"""Answer the question with a concise extract from the given context. Do not add any additional information, capital letters (only for names) or a punctuation mark in the end.\n\n
Here are some examples: \n\n
{few_shot_prompt}"""
     

# combine the systeminstruct + context + question into one column. This will be your input prompt.
test_df["systemInstruct"] = systemInstruct

test_df["input_question"] = (
    "\n\n **Below the question with context that you need to answer**"
    + "\n Context: "
    + test_df["context"]
    + "\n Question: "
    + test_df["question"]
)

test_systemInstruct = test_df["systemInstruct"].iloc[row_dataset]
print(test_systemInstruct)
test_question = test_df["input_question"].iloc[row_dataset]
print(test_question)
     

Next, set the model that you will use. In this example you will use "gemini-2.0-flash-001", a multimodal model that is designed for high-volume, cost-effective applications, and which delivers speed and efficiency to build fast, lower-cost applications that don't compromise on quality.

For the latest Gemini models and versions, please have a look at our documentation.

base_model = "gemini-2.0-flash-001"
     

y_true = test_df["answers"].values
y_pred_question = test_df["input_question"].values

# Check two pairs of question and answer.
for i in range(2):  # Loop through the first two indices
    print(f"Pair {i+1}:")
    print(f"  True Answer: {y_true[i]}")
    print(f"  Predicted Question: {y_pred_question[i]}")
     

Next lets take a question and get a prediction from Gemini that we can compare to the actual answer.

def get_predictions(question: str, model_version: str) -> str:

    prompt = question
    base_model = model_version

    response = client.models.generate_content(
        model=base_model,
        contents=prompt,
        config={
            "system_instruction": systemInstruct,
            "temperature": 0.3,
        },
    )

    return response.text
     

test_answer = test_df["answers"].iloc[row_dataset]
response = get_predictions(test_question, base_model)

print(f"Gemini response: {response}")
print(f"Actual answer: {test_answer}")
     

Sometimes you might get an answer from Gemini is more lengthy. However, answers in the SQuAD dataset are typically concise and clear.

Fine-tuning is a great way to control the type of output your use case requires. In this instance, you would want the model to provide short, clear answers.

# Apply the get_prediction() function to the 'question_column'
test_df["predicted_answer"] = test_df["input_question"].apply(get_predictions)
test_df.head(2)
     

You also need to make sure that the predicted answer is in the same format.

test_df["predicted_answer"] = test_df["predicted_answer"].apply(normalize_answer)
test_df.head(4)
     

Next, let's establish a baseline using evaluation metrics.

Evaluating the performance of a Question Answering (QA) system requires specific metrics. Two commonly used metrics are Exact Match (EM) and F1 score.

EM is a strict measure that only considers an answer correct if it perfectly matches the ground truth, even down to the punctuation. It's a binary metric - either 1 for a perfect match or 0 otherwise. This makes it sensitive to minor variations in phrasing.

F1 score is more flexible. It considers the overlap between the predicted answer and the true answer in terms of individual words or tokens. It calculates the harmonic mean of precision (proportion of correctly predicted words out of all predicted words) and recall (proportion of correctly predicted words out of all true answer words). This allows for partial credit and is less sensitive to minor wording differences.

In practice, EM is useful when exact wording is crucial, while F1 is more suitable when evaluating the overall understanding and semantic accuracy of the QA system. Often, both metrics are used together to provide a comprehensive evaluation.

def f1_score_squad(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def calculate_em_and_f1(y_true, y_pred):
    """Calculates EM and F1 scores for DataFrame columns."""

    # Ensure inputs are Series
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)

    em = np.mean(y_true.combine(y_pred, exact_match_score))
    f1 = np.mean(y_true.combine(y_pred, f1_score_squad))

    # # Print non-matching pairs (using index for clarity)
    # for i, (t, p) in enumerate(zip(y_true, y_pred)):
    #     if not exact_match_score(p, t):
    #         print(f"No EM Match at index {i}:\nTrue: {t}\nPred: {p}\n")

    return em, f1
     

em, f1 = calculate_em_and_f1(test_df["answers"], test_df["predicted_answer"])
print(f"EM score: {em}")
print(f"F1 score: {f1}")
     
Prepare the data for fine-tuning

To optimize the supervised fine-tuning process for a foundation model, ensure your dataset includes examples that reflect the desired task. Each record in the dataset pairs an input text (or prompt) with its corresponding expected output. This supervised tuning approach uses the dataset to effectively teach the model the specific behavior or task you need it to perform, by providing numerous illustrative examples.

The size of your dataset will vary depending on the complexity of the task, but as a general rule, the more examples you include, the better the model's performance. For fine-tuning Gemini on Vertex AI, the minimum number of examples are 100.
Dataset Format

Your training data should be structured in a JSONL file and stored at a Google Cloud Storage (GCS) URI. Each line in the JSONL file must adhere to the following schema:

A contents array containing objects that define:

    A role ("user" for user input or "model" for model output)
    parts containing the input data.

{
   "contents":[
      {
         "role":"user",  # This indicate input content
         "parts":[
            {
               "text":"How are you?"
            }
         ]
      },
      {
         "role":"model", # This indicate target content
         "parts":[ # text only
            {
               "text":"I am good, thank you!"
            }
         ]
      }
      #  ... repeat "user", "model" for multi turns.
   ]
}

Refer to the public documentation for more details.

# combine the systeminstruct + context + question into one column.
train_df = pd.read_csv("squad_train.csv")
validation_df = pd.read_csv("squad_validation.csv")
     

# combine the systeminstruct + context + question into one column.
train_df["input_question"] = (
    "\n\n **Below the question with context that you need to answer**"
    + "\n Context: "
    + train_df["context"]
    + "\n Question: "
    + train_df["question"]
)
validation_df["input_question"] = (
    "\n\n **Below the question with context that you need to answer**"
    + "\n Context: "
    + validation_df["context"]
    + "\n Question: "
    + validation_df["question"]
)
     

def df_to_jsonl(df, output_file):
    """Converts a Pandas DataFrame to JSONL format and saves it to a file.

    Args:
      df: The DataFrame to convert.
      output_file: The name of the output file.
    """

    with open(output_file, "w") as f:
        for row in df.itertuples(index=False):
            jsonl_obj = {
                "systemInstruction": {"parts": [{"text": f"{systemInstruct}"}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{row.input_question}"}],
                    },
                    {"role": "model", "parts": [{"text": row.answers}]},
                ],
            }
            f.write(json.dumps(jsonl_obj) + "\n")


# Process the DataFrames
df_to_jsonl(train_df, "squad_train.jsonl")
df_to_jsonl(validation_df, "squad_validation.jsonl")

print(f"JSONL data written to squad_train.jsonl")
print(f"JSONL data written to squad_validation.jsonl")
     

Next you will copy the files into your Google Cloud bucket

!gsutil cp ./squad_train.jsonl {BUCKET_URI}
!gsutil cp ./squad_validation.jsonl {BUCKET_URI}
     
Start fine-tuning job

Next you can start the fine-tuning job.

    source_model: Specifies the base Gemini model version you want to fine-tune.
    train_dataset: Path to your training data in JSONL format.

Optional parameters

    validation_dataset: If provided, this data is used to evaluate the model during tuning.
    tuned_model_display_name: Display name for the tuned model.
    epochs: The number of training epochs to run.
    learning_rate_multiplier: A value to scale the learning rate during training.
    adapter_size : Gemini 2.0 supports Adapter length [1, 4], default value is 4.

Important: The default hyperparameter settings are optimized for optimal performance based on rigorous testing and are recommended for initial use. Users may customize these parameters to address specific performance requirements.**

train_dataset = f"""{BUCKET_URI}/squad_train.jsonl"""
validation_dataset = f"""{BUCKET_URI}/squad_train.jsonl"""

training_dataset = {
    "gcs_uri": train_dataset,
}

validation_dataset = types.TuningValidationDataset(gcs_uri=validation_dataset)
     

sft_tuning_job = client.tunings.tune(
    base_model=base_model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        adapter_size="ADAPTER_SIZE_EIGHT",
        epoch_count=1,  # set to one to keep time and cost low
        tuned_model_display_name="gemini-flash-1.5-qa",
    ),
)
sft_tuning_job
     

Important: Tuning time depends on several factors, such as training data size, number of epochs, learning rate multiplier, etc.
⚠️ It will take ~30 mins for the model tuning job to complete on the provided dataset and set configurations/hyperparameters. ⚠️

sft_tuning_job.state
     

tuning_job = client.tunings.get(name=sft_tuning_job.name)
tuning_job
     
Model tuning metrics

    /train_total_loss: Loss for the tuning dataset at a training step.
    /train_fraction_of_correct_next_step_preds: The token accuracy at a training step. A single prediction consists of a sequence of tokens. This metric measures the accuracy of the predicted tokens when compared to the ground truth in the tuning dataset.
    /train_num_predictions: Number of predicted tokens at a training step

Model evaluation metrics:

    /eval_total_loss: Loss for the evaluation dataset at an evaluation step.
    /eval_fraction_of_correct_next_step_preds: The token accuracy at an evaluation step. A single prediction consists of a sequence of tokens. This metric measures the accuracy of the predicted tokens when compared to the ground truth in the evaluation dataset.
    /eval_num_predictions: Number of predicted tokens at an evaluation step.

The metrics visualizations are available after the model tuning job completes. If you don't specify a validation dataset when you create the tuning job, only the visualizations for the tuning metrics are available.

experiment_name = tuning_job.experiment
experiment_name
     

# Locate Vertex AI Experiment and Vertex AI Experiment Run
experiment = aiplatform.Experiment(experiment_name=experiment_name)
filter_str = metadata_utils._make_filter_string(
    schema_title="system.ExperimentRun",
    parent_contexts=[experiment.resource_name],
)
experiment_run = context.Context.list(filter_str)[0]
     

# Read data from Tensorboard
tensorboard_run_name = f"{experiment.get_backing_tensorboard_resource().resource_name}/experiments/{experiment.name}/runs/{experiment_run.name.replace(experiment.name, '')[1:]}"
tensorboard_run = aiplatform.TensorboardRun(tensorboard_run_name)
metrics = tensorboard_run.read_time_series_data()
     

def get_metrics(metric: str = "/train_total_loss"):
    """
    Get metrics from Tensorboard.

    Args:
      metric: metric name, eg. /train_total_loss or /eval_total_loss.
    Returns:
      steps: list of steps.
      steps_loss: list of loss values.
    """
    loss_values = metrics[metric].values
    steps_loss = []
    steps = []
    for loss in loss_values:
        steps_loss.append(loss.scalar.value)
        steps.append(loss.step)
    return steps, steps_loss
     

# Get Train and Eval Loss
train_loss = get_metrics(metric="/train_total_loss")
eval_loss = get_metrics(metric="/eval_total_loss")
     

# Plot the train and eval loss metrics using Plotly python library
fig = make_subplots(
    rows=1, cols=2, shared_xaxes=True, subplot_titles=("Train Loss", "Eval Loss")
)

# Add traces
fig.add_trace(
    go.Scatter(x=train_loss[0], y=train_loss[1], name="Train Loss", mode="lines"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=eval_loss[0], y=eval_loss[1], name="Eval Loss", mode="lines"),
    row=1,
    col=2,
)

# Add figure title
fig.update_layout(title="Train and Eval Loss", xaxis_title="Steps", yaxis_title="Loss")

# Set x-axis title
fig.update_xaxes(title_text="Steps")

# Set y-axes titles
fig.update_yaxes(title_text="Loss")

# Show plot
fig.show()
     
Use the fine-tuned model and evaluation

prompt = """
Answer the question based on the context

Context: In the 1840s and 50s, there were attempts to overcome this problem by means of various patent valve gears with a separate, variable cutoff expansion valve riding on the back of the main slide valve; the latter usually had fixed or limited cutoff.
The combined setup gave a fair approximation of the ideal events, at the expense of increased friction and wear, and the mechanism tended to be complicated.
The usual compromise solution has been to provide lap by lengthening rubbing surfaces of the valve in such a way as to overlap the port on the admission side, with the effect that the exhaust side remains open for a longer period after cut-off on the admission side has occurred.
This expedient has since been generally considered satisfactory for most purposes and makes possible the use of the simpler Stephenson, Joy and Walschaerts motions.
Corliss, and later, poppet valve gears had separate admission and exhaust valves driven by trip mechanisms or cams profiled so as to give ideal events; most of these gears never succeeded outside of the stationary marketplace due to various other issues including leakage and more delicate mechanisms.

Question: How is lap provided by overlapping the admission side port?
"""
     

tuned_model = tuning_job.tuned_model.endpoint
tuned_model
     

get_predictions(prompt, tuned_model)
     

# Apply the get_prediction() function to the 'question_column'
test_df["predicted_answer"] = test_df["input_question"].apply(get_predictions)
test_df.head(2)
     

test_df["predicted_answer"] = test_df["predicted_answer"].apply(normalize_answer)
     

After running the evaluation you can see that the model generally performs better on our use case after fine-tuning. Of course, depending on things like use case or data quality performance will differ.

em, f1 = calculate_em_and_f1(test_df["answers"], test_df["predicted_answer"])
print(f"EM score: {em}")
print(f"F1 score: {f1}")
     
