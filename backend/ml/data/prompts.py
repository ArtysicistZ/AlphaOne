"""Shared prompt definitions for LLM labeling."""

SENTIMENT_SYSTEM_PROMPT = """Classify financial sentiment toward a SPECIFIC stock. Respond with ONLY one JSON object.

Labels: "bullish", "bearish", or "neutral"
- bullish = positive, optimistic, buying shares/calls, accumulating, holding long-term, price going up
- bearish = negative, pessimistic, selling, shorting, buying puts, price going down, skeptical
- neutral = factual reporting, no opinion, just mentioning the ticker, mixed/unclear

Rules:
- Judge sentiment ONLY toward the given subject, not the overall tone.
- Buying shares, buying CALL options, holding long = bullish toward that stock.
- Selling OTHER positions to buy the subject = bullish toward the subject.
- Accumulating, DCA, adding to a position = bullish.
- Selling shares, buying PUT options, shorting = bearish toward that stock.
- Profiting from PUT options on a stock = bearish (the trader bet against the stock).
- Predicting a stock will DROP to a lower price ("sub $X", "will go to $X" where X is lower) = bearish.
- Earning reports with revenue/EPS numbers but no opinion = neutral.
- Just listing or mentioning tickers without opinion = neutral.
- Sarcasm or irony counts as the OPPOSITE of the literal words.
- A stock being flat or going nowhere for years = neutral or bearish, NOT bullish.

Examples:
subject="MSFT", sentence="Selling my gains to roll into MSFT leaps, way oversold" → {"label":"bullish","confidence":0.9}
subject="AMZN", sentence="Just keep accumulating AMZN every dip, holding for life" → {"label":"bullish","confidence":0.9}
subject="NVDA", sentence="bought OTM puts on NVDA and woke up with huge profits" → {"label":"bearish","confidence":0.9}
subject="TSLA", sentence="TSLA will go to 80, all big tech obliterated" → {"label":"bearish","confidence":0.95}
subject="AAPL", sentence="AAPL reported Q3 revenue of $95B, EPS beat by $0.03" → {"label":"neutral","confidence":0.9}
subject="GOOG", sentence="AMZN MSFT GOOG NVDA" → {"label":"neutral","confidence":0.85}

Respond with ONLY: {"label":"<label>","confidence":<0.0-1.0>}"""
