"""Universal what-if: one scenario engine, many domain packs.

    domains/   what each sector knows: concepts (columns it recognises), templates (questions it can ask), and which
               calculation answers each. A domain never calculates; it only describes.
    profile    DatasetProfile: what a table contains and which domain it looks like, with the evidence.
    discovery  suggested questions, each classified ready / needs assumptions / needs external data / unsupported.
    scenario   the standard scenario record and its validation.
    operations row-level calculations on a dataset (thresholds, capacity, association, trend).
    engine     runs a scenario: dataset operations here, formula models through app.scenario_service (the engine the
               business What-If page and the Copilot already share).
    explain    the eight-part explanation every result carries.
    service    the thin entry point the API router and the Copilot adapter both call.

No module here asks a language model for a number. A model may only pick among grounded templates (discovery).
"""
ENGINE_VERSION = "whatif.1"
