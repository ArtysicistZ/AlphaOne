SENTENCE_TOPIC_MAP = {
    # ── Top US Stocks by Market Cap ────────────────────────────────────
    #
    # REMOVED tickers whose uppercase KEY collides with common English
    # words in entity replacement (train_finbert.py _TICKER_PATTERN).
    # Single-letter keys (V, C, T) match standalone letters; common-word
    # keys (SO, NOW, LOW, PM, ICE, HD, MA, COP, USB) match normal text.
    # These create false [TARGET]/[OTHER] replacements that corrupt
    # training signal.  The removed tickers are rarely discussed by
    # ticker on Reddit anyway.
    #
    # Removed: V, C, T, SO, NOW, LOW, PM, ICE, HD, MA, COP, USB

    # Mega-cap Technology
    'AAPL': ['aapl', '$aapl', 'apple'],
    'MSFT': ['msft', '$msft', 'microsoft'],
    'NVDA': ['nvda', '$nvda', 'nvidia'],
    'AMZN': ['amzn', '$amzn', 'amazon'],
    'GOOG': ['goog', '$goog', 'googl', '$googl', 'google', 'alphabet'],
    'META': ['$meta', 'meta platforms', 'facebook'],
    'AVGO': ['avgo', '$avgo', 'broadcom'],
    'TSLA': ['tsla', '$tsla', 'tesla'],
    'ORCL': ['orcl', '$orcl', 'oracle'],
    'CRM': ['crm', '$crm', 'salesforce'],
    'AMD': ['amd', '$amd'],
    'ADBE': ['adbe', '$adbe', 'adobe'],
    'CSCO': ['csco', '$csco', 'cisco'],
    'IBM': ['ibm', '$ibm'],
    'INTU': ['intu', '$intu', 'intuit'],
    'QCOM': ['qcom', '$qcom', 'qualcomm'],
    'TXN': ['txn', '$txn', 'texas instruments'],
    'AMAT': ['amat', '$amat', 'applied materials'],
    'LRCX': ['lrcx', '$lrcx', 'lam research'],
    'KLAC': ['klac', '$klac'],
    'SNPS': ['snps', '$snps', 'synopsys'],
    'PANW': ['panw', '$panw', 'palo alto'],
    'INTC': ['intc', '$intc', 'intel'],
    'PLTR': ['pltr', '$pltr', 'palantir'],

    # Finance
    'BRK': ['brk', '$brk', 'berkshire', 'berkshire hathaway'],
    'JPM': ['jpm', '$jpm', 'jpmorgan', 'jp morgan'],
    'BAC': ['bac', '$bac', 'bank of america'],
    'WFC': ['wfc', '$wfc', 'wells fargo'],
    'GS': ['$gs', 'goldman', 'goldman sachs'],
    'MS': ['$ms', 'morgan stanley'],
    'SCHW': ['schw', '$schw', 'schwab', 'charles schwab'],
    'AXP': ['axp', '$axp', 'american express', 'amex'],
    'BLK': ['blk', '$blk', 'blackrock'],
    'SPGI': ['spgi', '$spgi', 's&p global'],
    'CME': ['cme', '$cme', 'cme group'],
    'CB': ['$cb', 'chubb'],
    'MMC': ['mmc', '$mmc', 'marsh mclennan'],
    'ADP': ['adp', '$adp'],
    'COIN': ['$coin', 'coinbase'],

    # Healthcare & Pharma
    'LLY': ['lly', '$lly', 'eli lilly'],
    'UNH': ['unh', '$unh', 'unitedhealth'],
    'JNJ': ['jnj', '$jnj', 'johnson & johnson', 'johnson and johnson'],
    'ABBV': ['abbv', '$abbv', 'abbvie'],
    'MRK': ['mrk', '$mrk', 'merck'],
    'TMO': ['tmo', '$tmo', 'thermo fisher'],
    'ABT': ['abt', '$abt', 'abbott'],
    'ISRG': ['isrg', '$isrg', 'intuitive surgical'],
    'DHR': ['dhr', '$dhr', 'danaher'],
    'PFE': ['pfe', '$pfe', 'pfizer'],
    'AMGN': ['amgn', '$amgn', 'amgen'],
    'BSX': ['bsx', '$bsx', 'boston scientific'],
    'SYK': ['syk', '$syk', 'stryker'],
    'BMY': ['bmy', '$bmy', 'bristol myers', 'bristol-myers'],
    'CI': ['$ci', 'cigna'],
    'ZTS': ['zts', '$zts', 'zoetis'],

    # Consumer
    'COST': ['$cost', 'costco'],
    'WMT': ['wmt', '$wmt', 'walmart'],
    'PG': ['$pg', 'procter & gamble', 'procter and gamble'],
    'NFLX': ['nflx', '$nflx', 'netflix'],
    'KO': ['$ko', 'coca-cola', 'coca cola'],
    'PEP': ['$pep', 'pepsi', 'pepsico'],
    'MCD': ['mcd', '$mcd', 'mcdonalds', "mcdonald's"],
    'BKNG': ['bkng', '$bkng', 'booking holdings', 'booking.com'],
    'MDLZ': ['mdlz', '$mdlz', 'mondelez'],
    'MO': ['$mo', 'altria'],
    'TGT': ['tgt', '$tgt', 'target corp', 'target corporation'],
    'CL': ['$cl', 'colgate'],
    'CMG': ['cmg', '$cmg', 'chipotle'],

    # Energy
    'XOM': ['xom', '$xom', 'exxon', 'exxonmobil'],
    'CVX': ['cvx', '$cvx', 'chevron'],
    'EOG': ['eog', '$eog', 'eog resources'],

    # Industrial
    'ACN': ['acn', '$acn', 'accenture'],
    'GE': ['$ge', 'general electric'],
    'CAT': ['$cat', 'caterpillar'],
    'RTX': ['rtx', '$rtx', 'raytheon'],
    'HON': ['$hon', 'honeywell'],
    'DE': ['$de', 'deere', 'john deere'],
    'UNP': ['unp', '$unp', 'union pacific'],
    'WM': ['$wm', 'waste management'],
    'LIN': ['$lin', 'linde'],

    # Utilities & Real Estate
    'NEE': ['$nee', 'nextera'],
    'DUK': ['duk', '$duk', 'duke energy'],
    'PLD': ['pld', '$pld', 'prologis'],
    'APD': ['apd', '$apd', 'air products'],

    # Telecom
    'VZ': ['$vz', 'verizon'],

    # Meme / Reddit-popular
    'GME': ['gme', '$gme', 'gamestop'],
    'AMC': ['amc', '$amc'],

    # Other notable
    'MCK': ['mck', '$mck', 'mckesson'],

    # ── General Topics ──────────────────────────────────────────────

    'MACRO': [
        'fed', 'inflation', 'interest rate', 'rates', 'economy', 'recession',
        'jpow', 'powell', 'cpi', 'gdp', 'unemployment', 'treasury',
        'bond', 'bonds', 'fomc', 'federal reserve', 'rate cut', 'rate hike',
        'tariff', 'tariffs', 'trade war', 'debt ceiling', 'fiscal',
    ],
    'TECHNOLOGY': [
        'tech', 'ai', 'artificial intelligence', 'cloud', 'semiconductor',
        'semiconductors', 'chip', 'chips', 'saas', 'cybersecurity', 'robotics',
    ],
}
