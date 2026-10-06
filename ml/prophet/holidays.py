"""
ml/prophet/holidays.py
───────────────────────
Germany Holiday Calendar Integration

Adds German federal holidays to a Prophet model instance using Prophet's
built-in country holiday support via the `holidays` library.

Regional selection justification:
  - The OPSD dataset target column is `DE_load_actual_entsoe_transparency`
    which represents Germany's national electricity load.
  - Germany (DE) is the correct holiday region.
  - Federal-level holidays only are included (regional/state holidays excluded).
  - This assumption is documented in the Model Card.

IMPORTANT: Do NOT apply a holiday calendar from an arbitrary country.
           The region must correspond to the dataset target region.
"""

import logging
from typing import Dict, Any

from prophet import Prophet

from ml.prophet.configuration import HOLIDAY_CALENDAR_DOC
from ml.configs.config import HOLIDAY_COUNTRY, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def add_holidays(model: Prophet, country: str = HOLIDAY_COUNTRY) -> Prophet:
    """
    Add country holiday calendar to a Prophet model instance.

    Parameters
    ----------
    model   : An unfitted Prophet model instance.
    country : ISO-3166 country code (default: "DE" for Germany).

    Returns
    -------
    Prophet model with holidays added (the same instance, modified in-place).
    """
    model.add_country_holidays(country_name=country)
    logger.info(
        "[HOLIDAYS] Added %s federal holiday calendar to Prophet model. "
        "Holiday version: %s",
        country,
        HOLIDAY_CALENDAR_DOC.get("version", "via prophet built-in"),
    )
    return model


def get_holiday_metadata() -> Dict[str, Any]:
    """Return holiday configuration metadata for artifact storage."""
    return {
        "holiday_country": HOLIDAY_COUNTRY,
        "holiday_version": HOLIDAY_CALENDAR_DOC.get("version"),
        "holiday_region": HOLIDAY_CALENDAR_DOC.get("region"),
        "holiday_includes": HOLIDAY_CALENDAR_DOC.get("includes"),
        "holiday_note": HOLIDAY_CALENDAR_DOC.get("note"),
    }
