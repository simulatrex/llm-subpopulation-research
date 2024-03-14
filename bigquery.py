import json
from typing import Iterable
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()


def fetch_current_progress(
    experiment_id: str,
    model_id: str,
    project_id: str,
    dataset_name: str,
    table_name: str,
):
    """
    This function fetches the current progress of an experiment from BigQuery.
    """

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    query = f"""
    SELECT character_id, price_tag, MAX(iteration_number) as max_iteration
    FROM `{table_id}`
    WHERE experiment_id = '{experiment_id}'
    AND model_id = '{model_id}'
    GROUP BY character_id, price_tag
    """
    query_job = client.query(query)
    results = query_job.result().to_dataframe()

    # Convert results into a dictionary for easier access
    progress_dict = {
        (row["character_id"], row["price_tag"]): row["max_iteration"]
        for _, row in results.iterrows()
    }

    return progress_dict


def fetch_current_seller_progress(
    experiment_id: str,
    model_id: str,
    project_id: str,
    dataset_name: str,
    table_name: str,
):
    """
    This function fetches the current progress of an experiment from BigQuery.
    """

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    query = f"""
    SELECT price_tag, MAX(iteration_number) as max_iteration
    FROM `{table_id}`
    WHERE experiment_id = '{experiment_id}'
    AND model_id = '{model_id}'
    GROUP BY price_tag
    """
    query_job = client.query(query)
    results = query_job.result().to_dataframe()

    # Convert results into a dictionary for easier access
    progress_dict = {
        (row["price_tag"]): row["max_iteration"] for _, row in results.iterrows()
    }

    return progress_dict


def fetch_current_iterations(
    character_id: str,
    experiment_id: str,
    price_tag: int,
    project_id: str,
    dataset_name: str,
    table_name: str,
    model_id: str,
):
    """
    This function fetches the current iterations for a specific character, experiment, and price tag from BigQuery.
    """

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    query = f"""
    SELECT iteration_number
    FROM `{table_id}`
    WHERE character_id = '{character_id}'
    AND experiment_id = '{experiment_id}'
    AND price_tag = {price_tag}
    AND model_id = '{model_id}'
    ORDER BY iteration_number DESC
    """
    try:
        query_job = client.query(query)
        results = query_job.result().to_dataframe()
        return results["iteration_number"].tolist()
    except Exception as e:
        # Log the error or print it out
        print(f"Failed to fetch current iterations: {e}")
        # Handle or re-raise the error appropriately
        raise


def fetch_current_seller_iterations(
    experiment_id: str,
    price_tag: int,
    project_id: str,
    dataset_name: str,
    table_name: str,
    model_id: str,
):
    """
    This function fetches the current iterations for a specific experiment, and price tag from BigQuery.
    """

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    query = f"""
    SELECT iteration_number
    FROM `{table_id}`
    WHERE experiment_id = '{experiment_id}'
    AND price_tag = {price_tag}
    AND model_id = '{model_id}'
    ORDER BY iteration_number DESC
    """
    try:
        query_job = client.query(query)
        results = query_job.result().to_dataframe()
        return results["iteration_number"].tolist()
    except Exception as e:
        # Log the error or print it out
        print(f"Failed to fetch current iterations: {e}")
        # Handle or re-raise the error appropriately
        raise


def save_single_experiment_result_to_bigquery(
    character_id: str,
    price_tag: str,
    iteration_number: int,
    llm_response: int,
    project_id: str,
    dataset_name: str,
    table_name: str,
    experiment_id: str,
    model_id: str,
):
    """
    This function saves a single experiment result to BigQuery.
    """

    price_int = (
        int(price_tag.replace("$", "").replace(",", ""))
        if isinstance(price_tag, str)
        else int(price_tag)
    )

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    # Define the unique key for the entry
    unique_key = {
        "character_id": character_id,
        "experiment_id": experiment_id,
        "iteration_number": iteration_number,
        "price_tag": price_int,
    }

    # Check if the entry exists
    existing_iterations = fetch_current_iterations(
        character_id=character_id,
        experiment_id=experiment_id,
        price_tag=price_int,
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
        model_id=model_id,
    )

    if iteration_number in existing_iterations:
        print(f"Entry for {unique_key} already exists. Skipping insertion.")
        return

    # Prepare data entry
    data_entry = {
        **unique_key,
        "llm_response": llm_response == 1,
        "stated_preference": llm_response == 1,
        "model_id": model_id,
        "experiment_date": datetime.now().date(),
    }

    # Convert data entry to DataFrame
    df_new = pd.DataFrame([data_entry])

    # Convert date objects to string (ISO format) in the DataFrame
    def convert_dates(value):
        if isinstance(value, date):
            return value.isoformat()
        return value

    rows_to_insert = df_new.applymap(convert_dates).to_dict(orient="records")

    # Write the DataFrame to BigQuery
    # try:
    #     job = client.load_table_from_dataframe(
    #         df_new,
    #         table_id,
    #         job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    #     )
    #     job.result()

    #     print(
    #         f"Data uploaded to BigQuery for experiment_id: {experiment_id} and character_id: {character_id}"
    #     )
    # except Exception as e:
    #     # Log the error or print it out
    #     print(f"Failed to write to BigQuery: {e}")
    #     # Handle or re-raise the error appropriately
    #     raise

    try:
        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors == []:
            print(
                f"Data uploaded to BigQuery for experiment_id: {experiment_id} and character_id: {character_id}"
            )
        else:
            print("Errors:")
            for error in errors:
                print(error)
    except Exception as e:
        print(f"Failed to write to BigQuery: {e}")
        raise


def save_single_seller_experiment_result_to_bigquery(
    price_tag: str,
    iteration_number: int,
    llm_response: int,
    project_id: str,
    dataset_name: str,
    table_name: str,
    experiment_id: str,
    model_id: str,
):
    """
    This function saves a single experiment result to BigQuery.
    """

    price_int = (
        int(price_tag.replace("$", "").replace(",", ""))
        if isinstance(price_tag, str)
        else int(price_tag)
    )

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    # Define the unique key for the entry
    unique_key = {
        "experiment_id": experiment_id,
        "iteration_number": iteration_number,
        "price_tag": price_int,
    }

    # Check if the entry exists
    existing_iterations = fetch_current_seller_iterations(
        experiment_id=experiment_id,
        price_tag=price_int,
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
        model_id=model_id,
    )

    if iteration_number in existing_iterations:
        print(f"Entry for {unique_key} already exists. Skipping insertion.")
        return

    # Prepare data entry
    data_entry = {
        **unique_key,
        "llm_response": llm_response == 1,
        "stated_preference": llm_response == 1,
        "model_id": model_id,
        "experiment_date": datetime.now().date(),
    }

    # Convert data entry to DataFrame
    df_new = pd.DataFrame([data_entry])

    # Convert date objects to string (ISO format) in the DataFrame
    def convert_dates(value):
        if isinstance(value, date):
            return value.isoformat()
        return value

    rows_to_insert = df_new.applymap(convert_dates).to_dict(orient="records")

    try:
        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors == []:
            print(f"Data uploaded to BigQuery for experiment_id: {experiment_id}")
        else:
            print("Errors:")
            for error in errors:
                print(error)
    except Exception as e:
        print(f"Failed to write to BigQuery: {e}")
        raise


def save_experiment_results_to_bigquery(
    results: dict,
    project_id: str,
    dataset_name: str,
    table_name: str,
    experiment_id: str,
    model_id: str,
):
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    # Step 1: Fetch Existing Data for specific experiment_id
    query = f"""
    SELECT character_id, experiment_id
    FROM `{table_id}`
    WHERE experiment_id = '{experiment_id}'
    AND model_id = '{model_id}'
    """
    query_job = client.query(query)
    existing_data = query_job.result().to_dataframe()

    # Step 2: Prepare New Data and Check for Duplicates
    new_data = []
    for (character_id, price_tag), values in results.items():
        price_int = (
            int(price_tag.replace("$", "").replace(",", ""))
            if isinstance(price_tag, str)
            else int(price_tag)
        )
        # Convert price_tag to integer or keep as string based on your requirement
        for iteration, value in enumerate(values):
            # Define a composite key for uniqueness
            unique_key = {
                "character_id": character_id,
                "experiment_id": experiment_id,
                "iteration_number": iteration + 1,  # Adjust if iteration starts at 0
                "price_tag": price_int,
            }

            # Check for duplicates based on this unique key
            if not existing_data[
                (existing_data.character_id == unique_key["character_id"])
                & (existing_data.experiment_id == unique_key["experiment_id"])
                & (existing_data.iteration_number == unique_key["iteration_number"])
                & (existing_data.price_tag == unique_key["price_tag"])
            ].empty:
                print(f"Duplicate entry found for {unique_key}. Skipping.")
                continue

            new_data.append(
                {
                    **unique_key,
                    "llm_response": value == 1,
                    "stated_preference": value == 1,
                    "model_id": model_id,
                    "experiment_date": datetime.now().date(),
                }
            )

    # Step 3: Convert new data to DataFrame
    df_new = pd.DataFrame(new_data)

    # Step 4: Write the DataFrame to BigQuery
    if not df_new.empty:
        job = client.load_table_from_dataframe(
            df_new,
            table_id,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        )
        job.result()
        print(f"New data uploaded to BigQuery for experiment_id: {experiment_id}")
    else:
        print("No new data to upload.")

    print("Operation Completed.")


def read_experiment_results(
    dataset_name: str,
    table_name: str,
    project_id: str,
    experiment_id: str,
    model_id: str,
    start_date: datetime.date,
    end_date: datetime.date,
):
    # Create a BigQuery client
    client = bigquery.Client(project=project_id)

    # Define table ID
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    # Read the table from BigQuery into a DataFrame with conditions on model_id and date range
    query = f"""
        SELECT * FROM {table_id}
        WHERE experiment_id = @experiment_id
        AND model_id = @model_id
        AND experiment_date BETWEEN @start_date AND @end_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("experiment_id", "STRING", experiment_id),
            bigquery.ScalarQueryParameter("model_id", "STRING", model_id),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    # Run the query with the job_config
    df = client.query(query, job_config=job_config).to_dataframe()

    return df


def get_grouped_experiment_results(
    dataset_name: str,
    table_name: str,
    project_id: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> Iterable[dict]:
    # Create a BigQuery client
    client = bigquery.Client(project=project_id)

    # Define table ID
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    # Read the table from BigQuery into a DataFrame with conditions on date range and group by model_id and experiment_id
    query = f"""
        SELECT * FROM {table_id}
        WHERE experiment_date BETWEEN @start_date AND @end_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    # Run the query with the job_config
    df = client.query(query, job_config=job_config).to_dataframe()

    # Group the DataFrame by model_id and experiment_id and convert to a list of dictionaries
    grouped_results = [
        {
            "model_id": model_id,
            "experiment_id": experiment_id,
            "records": records.to_dict("records"),
        }
        for (model_id, experiment_id), records in df.groupby(
            ["model_id", "experiment_id"]
        )
    ]

    return grouped_results
