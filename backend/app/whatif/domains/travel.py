"""Travel: trip budget, duration, accommodation, transport, daily expenses, itinerary. Runs on values you type; no live prices."""
from __future__ import annotations

from .base import HYPOTHETICAL, Concept, Domain, InputSpec, Template

CONCEPTS = (
    Concept("destination", "Destination", frozenset({"destination", "city", "country", "trip"}), "category", 1.0),
    Concept("nights", "Nights / days", frozenset({"nights", "days", "duration"}), "numeric", 1.0),
    Concept("lodging", "Accommodation cost", frozenset({"hotel", "lodging", "accommodation", "stay"}), "numeric", 1.5),
    Concept("transport", "Transport cost", frozenset({"flight", "fare", "transport", "taxi", "train", "airfare"}), "numeric", 1.5),
    Concept("food", "Food cost", frozenset({"food", "meals", "dining"}), "numeric", 1.0),
    Concept("date", "Date", frozenset({"date", "departure", "arrival"}), "datetime", 0.3),
)

TRIP = {
    "inputs": {"days": {"label": "Trip length", "unit": "days", "min": 1, "good": "up"},
               "hotel_per_night": {"label": "Hotel per night", "unit": "currency/days", "min": 0, "good": "down"},
               "food_per_day": {"label": "Food per day", "unit": "currency/days", "min": 0, "good": "down"},
               "local_transport_per_day": {"label": "Local transport per day", "unit": "currency/days", "min": 0, "good": "down"},
               "fixed_costs": {"label": "Flights and other fixed costs", "unit": "currency", "min": 0, "good": "down"},
               "destinations": {"label": "Destinations", "unit": "destinations", "min": 1},
               "cost_per_extra_destination": {"label": "Extra cost per destination after the first", "unit": "currency/destinations", "min": 0, "good": "down"}},
    "outputs": {"total_cost": {"label": "Total trip cost", "unit": "currency",
                               "formula": "days * (hotel_per_night + food_per_day + local_transport_per_day) + fixed_costs + (destinations - 1) * cost_per_extra_destination"},
                "cost_per_day": {"label": "Cost per day", "unit": "currency/days",
                                 "formula": "(days * (hotel_per_night + food_per_day + local_transport_per_day) + fixed_costs + (destinations - 1) * cost_per_extra_destination) / days"}},
    "primary_output": "total_cost"}

BASE = (InputSpec("days", "Trip length (days)", "days", None, 1, None, role="baseline"),
        InputSpec("hotel_per_night", "Hotel per night", "currency", None, 0, None, role="baseline"),
        InputSpec("food_per_day", "Food per day", "currency", None, 0, None, role="baseline"),
        InputSpec("local_transport_per_day", "Local transport per day", "currency", None, 0, None, role="baseline"),
        InputSpec("fixed_costs", "Flights and other fixed costs", "currency", None, 0, None, role="baseline"),
        InputSpec("destinations", "Destinations", "destinations", 1.0, 1, None, role="baseline"),
        InputSpec("cost_per_extra_destination", "Extra cost per added destination", "currency", 0.0, 0, None, role="baseline"))


def _t(id_, title, question, change_input, tags, patterns, reason):
    return Template(id=id_, title=title, question=question, method="deterministic", operation="formula_model", model=TRIP, manual=True, evidence=HYPOTHETICAL,
                    inputs=(*BASE, change_input), reason=reason, tags=frozenset(tags), patterns=patterns,
                    limitation="Uses the prices you type; it does not look up live hotel, flight or transport prices.")


EXTEND = _t("tr_extend", "Extend the trip", "What if I extend my trip by {extra_days} days?",
            InputSpec("extra_days", "Extra days", "days", None, None, None, role="change", target="days", change_type="abs"),
            ("extend", "longer", "days", "trip", "duration"), (r"extend\w*\s+(?:my\s+)?(?:trip|stay)", r"(?:more|extra|additional)\s+days", r"stay\s+\d+\s+(?:more|extra)"),
            "Daily costs scale with the length of the trip.")
HOTEL = _t("tr_hotel", "Hotel costs change", "What if hotel costs change by {hotel_change}%?",
           InputSpec("hotel_change", "Change in hotel price", "%", None, -100, 1000, role="change", target="hotel_per_night", change_type="pct"),
           ("hotel", "accommodation", "price"), (r"hotel\s+(?:costs?|prices?|rates?)", r"accommodation\s+(?:costs?|prices?)"), "Hotel is nights × the nightly price you give.")
TRANSPORT = _t("tr_transport", "Change local transport", "What if local transport costs change by {transport_change}%?",
               InputSpec("transport_change", "Change in local transport cost", "%", None, -100, 1000, role="change", target="local_transport_per_day", change_type="pct"),
               ("transport", "transportation", "taxi", "train"), (r"transport\w*", r"\btaxi\b|\btrain\b|\bbus\b"), "Local transport is a daily cost you give.")
DEST = _t("tr_destinations", "Visit more destinations", "What if I visit {extra_destinations} additional destinations?",
          InputSpec("extra_destinations", "Additional destinations", "destinations", None, 0, None, role="change", target="destinations", change_type="abs"),
          ("destinations", "visit", "itinerary", "cities"), (r"(?:visit|add)\s+\d*\s*(?:additional|more|extra)?\s*destinations?", r"\badditional\s+destinations?"),
          "Each destination after the first adds the extra cost you state.")
LIVE = Template(id="tr_live_prices", title="Actual live prices", question="What will flights and hotels actually cost on my dates?", method="deterministic",
                operation="none", external=("live flight and hotel prices",), reason="Listed so it is clear that prices are typed by you, not fetched.",
                tags=frozenset({"live", "actual", "cheapest", "flight", "flights", "availability"}), patterns=(r"cheapest", r"live\s+price", r"availability"),
                limitation="Silt has no verified live price source connected.")

TRAVEL = Domain(key="travel", title="Travel", description="Trip budget, duration, accommodation, transport and itinerary. Works from values you type.",
                entities=("trip", "destination"), concepts=CONCEPTS, templates=(EXTEND, HOTEL, TRANSPORT, DEST, LIVE),
                metrics=("total trip cost", "cost per day"), keywords=frozenset({"itinerary", "trip", "travel", "tour", "booking"}),
                notes=("Prices are the ones you enter. Silt does not look up live prices.",))
