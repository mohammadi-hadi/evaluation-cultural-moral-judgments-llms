"""
Data processing module for loading and preprocessing WVS and PEW survey data.
"""

from typing import Dict, List, Optional

import pandas as pd
import pyreadstat

# Constants for WVS
COUNTRIES_WVS_W7_ALL = [
    "Andorra",
    "Argentina",
    "Armenia",
    "Australia",
    "Bangladesh",
    "Bolivia",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Colombia",
    "Cyprus",
    "Ecuador",
    "Egypt",
    "Ethiopia",
    "Germany",
    "Greece",
    "Guatemala",
    "Indonesia",
    "Iran",
    "Iraq",
    "Japan",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kyrgyzstan",
    "Lebanon",
    "Libya",
    "Malaysia",
    "Maldives",
    "Mexico",
    "Mongolia",
    "Morocco",
    "Myanmar",
    "Netherlands",
    "New Zealand",
    "Nicaragua",
    "Nigeria",
    "Pakistan",
    "Peru",
    "Philippines",
    "Romania",
    "Russia",
    "Singapore",
    "South Korea",
    "Taiwan ROC",
    "Tajikistan",
    "Thailand",
    "Tunisia",
    "Turkey",
    "Ukraine",
    "United States",
    "Venezuela",
    "Vietnam",
    "Zimbabwe",
]

W7_QUESTIONS = ["Q" + str(i) for i in range(177, 196)]
W7_QUESTIONS_TEXT = [
    "claiming government benefits to which you are not entitled",
    "avoiding a fare on public transport",
    "stealing property",
    "cheating on taxes",
    "someone accepting a bribe in the course of their duties",
    "homosexuality",
    "prostitution",
    "abortion",
    "divorce",
    "sex before marriage",
    "suicide",
    "euthanasia",
    "for a man to beat his wife",
    "parents beating children",
    "violence against other people",
    "terrorism as a political, ideological or religious mean",
    "having casual sex",
    "political violence",
    "death penalty",
]

# Constants for PEW
COUNTRIES_PEW_ALL = [
    "United States",
    "Czech Republic",
    "South Korea",
    "Canada",
    "France",
    "Germany",
    "Spain",
    "Mexico",
    "Chile",
    "Australia",
    "Russia",
    "Britain",
    "Turkey",
    "Greece",
    "Egypt",
    "Poland",
    "Senegal",
    "Italy",
    "Brazil",
    "Lebanon",
    "Nigeria",
    "Japan",
    "Malaysia",
    "Kenya",
    "Indonesia",
    "Uganda",
    "Jordan",
    "Argentina",
    "Philippines",
    "Tunisia",
    "China",
    "Pakistan",
    "Ghana",
    "South Africa",
    "Palestinian territories",
    "Israel",
    "Bolivia",
    "Venezuela",
    "El Salvador",
]

PEW_QUESTIONS = ["Q84" + chr(i) for i in range(ord("A"), ord("H") + 1)]
PEW_QUESTIONS_TEXT = [
    "using contraceptives",
    "getting a divorce",
    "having an abortion",
    "homosexuality",
    "drinking alcohol",
    "married people having an affair",
    "gambling",
    "sex between unmarried adults",
]

# Scaling constants for WVS
WVS_MINUS = 5.5
WVS_DIVIDE = 4.5


def load_wvs_data(filepath: str, country_codes_filepath: str = None) -> pd.DataFrame:
    """
    Load WVS moral questions data and join with country names.

    Args:
        filepath: Path to WVS_Moral.csv
        country_codes_filepath: Path to Country_Codes_Names.csv

    Returns:
        DataFrame with WVS data including country names
    """
    wvs_df = pd.read_csv(filepath)

    if country_codes_filepath:
        country_names_df = pd.read_csv(country_codes_filepath)
        wvs_df = wvs_df.set_index("B_COUNTRY").join(
            country_names_df.set_index("B_COUNTRY"), how="left"
        )

    return wvs_df


def load_pew_data(filepath: str) -> pd.DataFrame:
    """
    Load PEW dataset from SPSS file and preprocess.

    Args:
        filepath: Path to PEW .sav file

    Returns:
        DataFrame with processed PEW data
    """
    # Read SPSS file
    pew_data_original, meta = pyreadstat.read_sav(filepath)

    # Filter relevant columns
    filtered_columns = pew_data_original.filter(regex="^Q84[A-H]|COUNTRY").copy()
    filtered_columns.rename(columns={"COUNTRY": "Country_Names"}, inplace=True)

    # Replace textual answers with numeric codes
    replace_map = {
        "Morally acceptable": 1,
        "Not a moral issue": 0,
        "Morally unacceptable": -1,
        "Depends on situation (Volunteered)": 0,
        "Refused": 0,
        "Don't know": 0,
    }
    filtered_columns.replace(replace_map, inplace=True)

    # Convert to numeric
    for col in filtered_columns.columns[1:]:
        filtered_columns[col] = pd.to_numeric(filtered_columns[col], errors="coerce")

    return filtered_columns


def get_wvs_ratings(
    wvs_df: pd.DataFrame, culture: str, question: str
) -> Optional[float]:
    """
    Get mean rating for a specific question and culture in WVS data.
    Scales from original 1-10 scale to -1 to 1.

    Args:
        wvs_df: WVS DataFrame
        culture: Country name
        question: Question code (e.g., 'Q177')

    Returns:
        Scaled mean rating or None if not available
    """
    df = wvs_df[["Country_Names", question]].loc[wvs_df["Country_Names"] == culture]
    if df.empty:
        return None

    ratings = df.loc[df[question] > 0, question]
    if ratings.empty:
        return None

    # Scale from 1-10 to -1 to 1
    scaled = ((ratings - WVS_MINUS) / WVS_DIVIDE).mean()
    return scaled


def get_pew_ratings(
    pew_df: pd.DataFrame, culture: str, question: str
) -> Optional[float]:
    """
    Get mean rating for a specific question and culture in PEW data.

    Args:
        pew_df: PEW DataFrame
        culture: Country name
        question: Question code (e.g., 'Q84A')

    Returns:
        Mean rating or None if not available
    """
    df = pew_df[["Country_Names", question]]
    df = df.loc[df["Country_Names"] == culture]

    if df.empty:
        return None

    mean_rating = df[question].mean()
    if pd.isna(mean_rating):
        return None

    return mean_rating


def get_question_mapping(dataset: str) -> Dict[str, str]:
    """
    Get mapping of question codes to question text.

    Args:
        dataset: Either 'wvs' or 'pew'

    Returns:
        Dictionary mapping question codes to text
    """
    if dataset.lower() == "wvs":
        return dict(zip(W7_QUESTIONS, W7_QUESTIONS_TEXT))
    elif dataset.lower() == "pew":
        return dict(zip(PEW_QUESTIONS, PEW_QUESTIONS_TEXT))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_all_countries(dataset: str) -> List[str]:
    """
    Get list of all countries for a dataset.

    Args:
        dataset: Either 'wvs' or 'pew'

    Returns:
        List of country names
    """
    if dataset.lower() == "wvs":
        return COUNTRIES_WVS_W7_ALL
    elif dataset.lower() == "pew":
        return COUNTRIES_PEW_ALL
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
