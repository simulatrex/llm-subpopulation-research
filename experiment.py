import time
from typing import List, Dict
from pydantic import BaseModel
from requests import HTTPError
from bigquery import (
    fetch_current_progress,
    fetch_current_seller_progress,
    save_single_experiment_result_to_bigquery,
    save_single_seller_experiment_result_to_bigquery,
)

from llm_utils import CognitiveModel, OpenAILanguageModel, OpenRouterProxyLanguageModel


def run_llm_experiment(
    price_tags: List[str],
    characters: Dict,
    iterations: int,
    llm_model: CognitiveModel,
    prompts: Dict,
    experiment_id: str,
    project_id: str,
    dataset_name: str,
    table_name: str,
    debug: bool = False,
):
    """
    This function runs an experiment using the OpenRouter API.
    It will retry even if a 504 HTTP error is encountered from the model.
    """
    # Initialize the language model
    llm = OpenRouterProxyLanguageModel(model_id=llm_model)

    # Fetch current progress from the database
    current_progress = fetch_current_progress(
        experiment_id=experiment_id,
        model_id=llm_model.value,
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
    )

    # Loop over the price tags and character IDs
    for price_tag in price_tags:
        for character_id in characters.keys():
            price_tag_int = int(price_tag.replace("$", "").replace(",", ""))

            # Determine the last completed iteration for this character and price tag
            last_completed_iteration = current_progress.get(
                (character_id, price_tag_int), -1
            )

            if debug:
                print(
                    f"Last completed iteration for character {character_id} and price tag {price_tag}: {last_completed_iteration}"
                )

            # If no progress was found, start from 0; otherwise, continue from the next iteration.
            start_iteration = (
                0 if last_completed_iteration == -1 else last_completed_iteration + 1
            )

            # Continue data collection from the last completed iteration or start anew if none found
            for iteration in range(start_iteration, iterations):
                character_desc = characters[character_id]
                prompt = f"{prompts['default']} {character_desc} {prompts['experiment'].format(price_tag=price_tag)}"

                response = None
                while response is None:
                    try:
                        # Generate response (assuming llm.ask() is a method to get responses)
                        response = llm.ask(prompt)
                    except HTTPError as e:
                        if e.response.status_code == 504:
                            # Log the error and retry
                            if debug:
                                print(
                                    f"HTTP 504 Error encountered. Retrying: {character_id}, {price_tag}, {iteration}"
                                )
                            time.sleep(5)  # Wait a bit before retrying
                        else:
                            # If it's not a 504 error, raise the exception
                            raise

                if debug:
                    print(
                        f"Debug: {character_id}, {price_tag}, {iteration}, {response}"
                    )

                answer = response.strip()
                llm_response = 1 if "yes" in answer.lower() else 0

                # Save results
                save_single_experiment_result_to_bigquery(
                    character_id=character_id,
                    price_tag=price_tag,
                    iteration_number=iteration,
                    llm_response=llm_response,
                    project_id=project_id,
                    dataset_name=dataset_name,
                    table_name=table_name,
                    experiment_id=experiment_id,
                    model_id=llm_model.value,
                )

                time.sleep(5)  # Sleep between iterations


def run_llm_seller_experiment(
    price_tags: List[str],
    iterations: int,
    llm_model: CognitiveModel,
    prompts: Dict,
    experiment_id: str,
    project_id: str,
    dataset_name: str,
    table_name: str,
    debug: bool = False,
):
    """
    This function runs an experiment using the OpenAI API.
    """
    # Initialize the language model
    llm = OpenRouterProxyLanguageModel(model_id=llm_model)

    # Fetch current progress from the database
    current_progress = fetch_current_seller_progress(
        experiment_id=experiment_id,
        model_id=llm_model.value,
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
    )

    # Loop over the price tags and character IDs
    for price_tag in price_tags:
        price_tag_int = int(price_tag.replace("$", "").replace(",", ""))

        # Determine the last completed iteration for this character and price tag
        last_completed_iteration = current_progress.get((price_tag_int), -1)

        if debug:
            print(
                f"Last completed iteration for price tag {price_tag}: {last_completed_iteration}"
            )

        # If no progress was found, start from 0; otherwise, continue from the next iteration.
        start_iteration = (
            0 if last_completed_iteration == -1 else last_completed_iteration + 1
        )

        # Continue data collection from the last completed iteration or start anew if none found
        for iteration in range(start_iteration, iterations):
            prompt = f"{prompts['default']} {prompts['experiment'].format(price_tag=price_tag)}"

            # Generate response (assuming llm.ask() is a method to get responses)
            # TODO: use functions for opensource models
            response = llm.ask(prompt)

            if debug:
                print(f"Debug: {price_tag}, {iteration}, {response}")

            answer = response.strip()
            llm_response = 1 if "yes" in answer.lower() else 0

            # Save results
            save_single_seller_experiment_result_to_bigquery(
                price_tag=price_tag,
                iteration_number=iteration,
                llm_response=llm_response,
                project_id=project_id,
                dataset_name=dataset_name,
                table_name=table_name,
                experiment_id=experiment_id,
                model_id=llm_model.value,
            )

            time.sleep(0.5)  # Sleep between iterations


def run_openai_llm_experiment(
    price_tags: List[str],
    characters: Dict,
    iterations: int,
    llm_model: CognitiveModel,
    prompts: Dict,
    experiment_id: str,
    project_id: str,
    dataset_name: str,
    table_name: str,
    debug: bool = False,
):
    """
    This function runs an experiment using the OpenAI API.
    """
    # Initialize the language model
    llm = OpenAILanguageModel(model_id=llm_model)

    # Fetch current progress from the database
    current_progress = fetch_current_progress(
        experiment_id=experiment_id,
        model_id=llm_model.value,
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
    )

    class CharacterResponse(BaseModel):
        would_buy: bool

    # Loop over the price tags and character IDs
    for price_tag in price_tags:
        for character_id in characters.keys():
            price_tag_int = int(price_tag.replace("$", "").replace(",", ""))

            # Determine the last completed iteration for this character and price tag
            last_completed_iteration = current_progress.get(
                (character_id, price_tag_int), -1
            )

            if debug:
                print(
                    f"Last completed iteration for character {character_id} and price tag {price_tag}: {last_completed_iteration}"
                )

            # If no progress was found, start from 0; otherwise, continue from the next iteration.
            start_iteration = (
                0 if last_completed_iteration == -1 else last_completed_iteration + 1
            )

            # Continue data collection from the last completed iteration or start anew if none found
            for iteration in range(start_iteration, iterations):
                character_desc = characters[character_id]
                prompt = f"{prompts['default']} {character_desc} {prompts['experiment'].format(price_tag=price_tag)}"

                # Generate response (assuming llm.ask() is a method to get responses)
                response = llm.generate_structured_output(prompt, CharacterResponse)

                if debug:
                    print(
                        f"Debug: {character_id}, {price_tag}, {iteration}, {response}"
                    )

                llm_response = 1 if response.would_buy else 0

                # Save results
                save_single_experiment_result_to_bigquery(
                    character_id=character_id,
                    price_tag=price_tag,
                    iteration_number=iteration,
                    llm_response=llm_response,
                    project_id=project_id,
                    dataset_name=dataset_name,
                    table_name=table_name,
                    experiment_id=experiment_id,
                    model_id=llm_model.value,
                )

                time.sleep(0.5)  # Sleep between iterations


def run_openai_llm_seller_experiment(
    price_tags: List[str],
    iterations: int,
    llm_model: CognitiveModel,
    prompts: Dict,
    experiment_id: str,
    project_id: str,
    dataset_name: str,
    table_name: str,
    debug: bool = False,
):
    """
    This function runs an experiment using the OpenAI API.
    """
    # Initialize the language model
    llm = OpenAILanguageModel(model_id=llm_model)

    # Fetch current progress from the database
    current_progress = fetch_current_seller_progress(
        experiment_id=experiment_id,
        model_id=llm_model.value,
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
    )

    class CharacterResponse(BaseModel):
        would_buy: bool

    # Loop over the price tags and character IDs
    for price_tag in price_tags:
        price_tag_int = int(price_tag.replace("$", "").replace(",", ""))

        # Determine the last completed iteration for this character and price tag
        last_completed_iteration = current_progress.get((price_tag_int), -1)

        if debug:
            print(
                f"Last completed iteration for price tag {price_tag}: {last_completed_iteration}"
            )

        # If no progress was found, start from 0; otherwise, continue from the next iteration.
        start_iteration = (
            0 if last_completed_iteration == -1 else last_completed_iteration + 1
        )

        # Continue data collection from the last completed iteration or start anew if none found
        for iteration in range(start_iteration, iterations):
            prompt = f"{prompts['default']} {prompts['experiment'].format(price_tag=price_tag)}"

            # Generate response (assuming llm.ask() is a method to get responses)
            response = llm.generate_structured_output(prompt, CharacterResponse)

            if debug:
                print(f"Debug: {price_tag}, {iteration}, {response}")

            llm_response = 1 if response.would_buy else 0

            # Save results
            save_single_seller_experiment_result_to_bigquery(
                price_tag=price_tag,
                iteration_number=iteration,
                llm_response=llm_response,
                project_id=project_id,
                dataset_name=dataset_name,
                table_name=table_name,
                experiment_id=experiment_id,
                model_id=llm_model.value,
            )

            time.sleep(0.5)  # Sleep between iterations
