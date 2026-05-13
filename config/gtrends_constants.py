"""
Valid Google Trends dropdown values extracted from trends.google.com source.
Use these for strict validation during OCR metadata extraction.

Source: Google Trends HTML/JS (parsed from <script> variables)
- geoPicker: 48 primary countries shown in main dropdown
- allCountriesGeoPicker: 250+ countries/territories for edge cases
- exploreTimePicker: 10 official time range options
- catPicker: 7 category filters
- localePicker: 6 supported interface languages
"""

# =============================================================================
# PRIMARY COUNTRY DROPDOWN (geoPicker) - 48 countries shown in main UI
# =============================================================================
GEO_PICKER = {
    "AR": "Argentina",
    "AU": "Australia",
    "AT": "Austria",
    "BE": "Belgium",
    "BR": "Brazil",
    "CA": "Canada",
    "CL": "Chile",
    "CO": "Colombia",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EG": "Egypt",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "HK": "Hong Kong",
    "HU": "Hungary",
    "IN": "India",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IL": "Israel",
    "IT": "Italy",
    "JP": "Japan",
    "KE": "Kenya",
    "MY": "Malaysia",
    "MX": "Mexico",
    "NL": "Netherlands",
    "NZ": "New Zealand",
    "NG": "Nigeria",
    "NO": "Norway",
    "PE": "Peru",
    "PH": "Philippines",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "RU": "Russia",
    "SA": "Saudi Arabia",
    "SG": "Singapore",
    "ZA": "South Africa",
    "KR": "South Korea",
    "ES": "Spain",
    "SE": "Sweden",
    "CH": "Switzerland",
    "TW": "Taiwan",
    "TH": "Thailand",
    "TR": "Türkiye",
    "UA": "Ukraine",
    "GB": "United Kingdom",
    "US": "United States",
    "VN": "Vietnam",
}

# =============================================================================
# FULL COUNTRY LIST (allCountriesGeoPicker) - 250+ entries for edge cases
# Includes territories, dependencies, and special administrative regions
# =============================================================================
ALL_COUNTRIES = {
    "AF": "Afghanistan",
    "AX": "Åland Islands",
    "AL": "Albania",
    "DZ": "Algeria",
    "AS": "American Samoa",
    "AD": "Andorra",
    "AO": "Angola",
    "AI": "Anguilla",
    "AQ": "Antarctica",
    "AG": "Antigua & Barbuda",
    "AR": "Argentina",
    "AM": "Armenia",
    "AW": "Aruba",
    "AU": "Australia",
    "AT": "Austria",
    "AZ": "Azerbaijan",
    "BS": "Bahamas",
    "BH": "Bahrain",
    "BD": "Bangladesh",
    "BB": "Barbados",
    "BY": "Belarus",
    "BE": "Belgium",
    "BZ": "Belize",
    "BJ": "Benin",
    "BM": "Bermuda",
    "BT": "Bhutan",
    "BO": "Bolivia",
    "BA": "Bosnia & Herzegovina",
    "BW": "Botswana",
    "BV": "Bouvet Island",
    "BR": "Brazil",
    "IO": "British Indian Ocean Territory",
    "VG": "British Virgin Islands",
    "BN": "Brunei",
    "BG": "Bulgaria",
    "BF": "Burkina Faso",
    "BI": "Burundi",
    "KH": "Cambodia",
    "CM": "Cameroon",
    "CA": "Canada",
    "CV": "Cape Verde",
    "BQ": "Caribbean Netherlands",
    "KY": "Cayman Islands",
    "CF": "Central African Republic",
    "TD": "Chad",
    "CL": "Chile",
    "CN": "China",
    "CX": "Christmas Island",
    "CC": "Cocos (Keeling) Islands",
    "CO": "Colombia",
    "KM": "Comoros",
    "CG": "Congo - Brazzaville",
    "CD": "Congo - Kinshasa",
    "CK": "Cook Islands",
    "CR": "Costa Rica",
    "CI": "Côte d'Ivoire",
    "HR": "Croatia",
    "CU": "Cuba",
    "CW": "Curaçao",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DK": "Denmark",
    "DJ": "Djibouti",
    "DM": "Dominica",
    "DO": "Dominican Republic",
    "EC": "Ecuador",
    "EG": "Egypt",
    "SV": "El Salvador",
    "GQ": "Equatorial Guinea",
    "ER": "Eritrea",
    "EE": "Estonia",
    "SZ": "Eswatini",
    "ET": "Ethiopia",
    "FK": "Falkland Islands (Islas Malvinas)",
    "FO": "Faroe Islands",
    "FJ": "Fiji",
    "FI": "Finland",
    "FR": "France",
    "GF": "French Guiana",
    "PF": "French Polynesia",
    "TF": "French Southern Territories",
    "GA": "Gabon",
    "GM": "Gambia",
    "GE": "Georgia",
    "DE": "Germany",
    "GH": "Ghana",
    "GI": "Gibraltar",
    "GR": "Greece",
    "GL": "Greenland",
    "GD": "Grenada",
    "GP": "Guadeloupe",
    "GU": "Guam",
    "GT": "Guatemala",
    "GG": "Guernsey",
    "GN": "Guinea",
    "GW": "Guinea-Bissau",
    "GY": "Guyana",
    "HT": "Haiti",
    "HM": "Heard & McDonald Islands",
    "HN": "Honduras",
    "HK": "Hong Kong",
    "HU": "Hungary",
    "IS": "Iceland",
    "IN": "India",
    "ID": "Indonesia",
    "IR": "Iran",
    "IQ": "Iraq",
    "IE": "Ireland",
    "IM": "Isle of Man",
    "IL": "Israel",
    "IT": "Italy",
    "JM": "Jamaica",
    "JP": "Japan",
    "JE": "Jersey",
    "JO": "Jordan",
    "KZ": "Kazakhstan",
    "KE": "Kenya",
    "KI": "Kiribati",
    "XK": "Kosovo",
    "KW": "Kuwait",
    "KG": "Kyrgyzstan",
    "LA": "Laos",
    "LV": "Latvia",
    "LB": "Lebanon",
    "LS": "Lesotho",
    "LR": "Liberia",
    "LY": "Libya",
    "LI": "Liechtenstein",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MO": "Macao",
    "MG": "Madagascar",
    "MW": "Malawi",
    "MY": "Malaysia",
    "MV": "Maldives",
    "ML": "Mali",
    "MT": "Malta",
    "MH": "Marshall Islands",
    "MQ": "Martinique",
    "MR": "Mauritania",
    "MU": "Mauritius",
    "YT": "Mayotte",
    "MX": "Mexico",
    "FM": "Micronesia",
    "MD": "Moldova",
    "MC": "Monaco",
    "MN": "Mongolia",
    "ME": "Montenegro",
    "MS": "Montserrat",
    "MA": "Morocco",
    "MZ": "Mozambique",
    "MM": "Myanmar (Burma)",
    "NA": "Namibia",
    "NR": "Nauru",
    "NP": "Nepal",
    "NL": "Netherlands",
    "NC": "New Caledonia",
    "NZ": "New Zealand",
    "NI": "Nicaragua",
    "NE": "Niger",
    "NG": "Nigeria",
    "NU": "Niue",
    "NF": "Norfolk Island",
    "KP": "North Korea",
    "MK": "North Macedonia",
    "MP": "Northern Mariana Islands",
    "NO": "Norway",
    "OM": "Oman",
    "PK": "Pakistan",
    "PW": "Palau",
    "PS": "Palestine",
    "PA": "Panama",
    "PG": "Papua New Guinea",
    "PY": "Paraguay",
    "PE": "Peru",
    "PH": "Philippines",
    "PN": "Pitcairn Islands",
    "PL": "Poland",
    "PT": "Portugal",
    "PR": "Puerto Rico",
    "QA": "Qatar",
    "RE": "Réunion",
    "RO": "Romania",
    "RU": "Russia",
    "RW": "Rwanda",
    "WS": "Samoa",
    "SM": "San Marino",
    "ST": "São Tomé & Príncipe",
    "SA": "Saudi Arabia",
    "SN": "Senegal",
    "RS": "Serbia",
    "SC": "Seychelles",
    "SL": "Sierra Leone",
    "SG": "Singapore",
    "SX": "Sint Maarten",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "SB": "Solomon Islands",
    "SO": "Somalia",
    "ZA": "South Africa",
    "GS": "South Georgia & South Sandwich Islands",
    "KR": "South Korea",
    "SS": "South Sudan",
    "ES": "Spain",
    "LK": "Sri Lanka",
    "BL": "St Barthélemy",
    "SH": "St Helena",
    "KN": "St Kitts & Nevis",
    "LC": "St Lucia",
    "MF": "St Martin",
    "PM": "St Pierre & Miquelon",
    "VC": "St Vincent & the Grenadines",
    "SD": "Sudan",
    "SR": "Suriname",
    "SJ": "Svalbard & Jan Mayen",
    "SE": "Sweden",
    "CH": "Switzerland",
    "SY": "Syria",
    "TW": "Taiwan",
    "TJ": "Tajikistan",
    "TZ": "Tanzania",
    "TH": "Thailand",
    "TL": "Timor-Leste",
    "TG": "Togo",
    "TK": "Tokelau",
    "TO": "Tonga",
    "TT": "Trinidad & Tobago",
    "TN": "Tunisia",
    "TR": "Türkiye",
    "TM": "Turkmenistan",
    "TC": "Turks & Caicos Islands",
    "TV": "Tuvalu",
    "UG": "Uganda",
    "UA": "Ukraine",
    "AE": "United Arab Emirates",
    "GB": "United Kingdom",
    "US": "United States",
    "UY": "Uruguay",
    "UM": "US Outlying Islands",
    "VI": "US Virgin Islands",
    "UZ": "Uzbekistan",
    "VU": "Vanuatu",
    "VA": "Vatican City",
    "VE": "Venezuela",
    "VN": "Vietnam",
    "WF": "Wallis & Futuna",
    "EH": "Western Sahara",
    "YE": "Yemen",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
}

# =============================================================================
# TIME RANGE DROPDOWN (exploreTimePicker) - 10 official options
# =============================================================================
TIME_RANGES = {
    "now 1-H": "Past hour",
    "now 4-H": "Past 4 hours",
    "now 1-d": "Past day",
    "now 7-d": "Past 7 days",
    "today 1-m": "Past 30 days",
    "today 3-m": "Past 90 days",
    "today 12-m": "Past 12 months",
    "today 5-y": "Past 5 years",
    "all_2008": "2008 – present",
    "all": "2004 – present",
}

# =============================================================================
# CATEGORY FILTERS (catPicker) - 7 options
# =============================================================================
CATEGORIES = {
    "all": "All categories",
    "b": "Business",
    "e": "Entertainment",
    "m": "Health",
    "t": "Sci/Tech",
    "s": "Sports",
    "h": "Top stories",
}

# =============================================================================
# SUPPORTED LOCALES (localePicker) - 6 interface languages
# =============================================================================
LOCALES = {
    "en_US": "English (United States)",
    "en_GB": "English (United kingdom)",
    "ar": "العربية",
    "iw": "עברית",
    "zh_CN": "中文（中国）",
    "zh_TW": "中文（台灣）",
}

# =============================================================================
# UI KEYWORDS TO FILTER OUT DURING OCR PARSING
# These appear in GTrends interface but are not user search terms
# =============================================================================
GTRENDS_UI_KEYWORDS = {
    # Main navigation
    "google trends",
    "explore",
    "trending now",
    "year in search",
    # Chart labels
    "search interest",
    "interest over time",
    "interest by subregion",
    "related queries",
    "related topics",
    # Filter labels
    "time range",
    "location",
    "country",
    "region",
    "city",
    "category",
    "web search",
    "image search",
    "youtube search",
    "news",
    "google shopping",
    # Data states
    "data is limited",
    "no data available",
    "compare",
    "rising",
    "top",
    "breakout",
    # Time modifiers
    "realtime",
    "past hour",
    "past 4 hours",
    "past day",
    "past 7 days",
    "past 30 days",
    "past 90 days",
    "past 12 months",
    "past 5 years",
    "2004 – present",
    "2008 – present",
    # Actions
    "download",
    "embed",
    "share",
    "subscribe",
    # Misc
    "google",
    "trends",
    "worldwide",
    "all categories",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_country_name(code: str, fallback_to_all: bool = True) -> str:
    """
    Get human-readable country name from ISO code.
    
    Args:
        code: Two-letter ISO country code (e.g., "US", "GB")
        fallback_to_all: If True, search ALL_COUNTRIES if not in GEO_PICKER
    
    Returns:
        Country name string, or the original code if not found
    """
    code_upper = code.upper().split("-")[0]  # Handle subregions like "BR-DF"
    
    # Check primary list first (faster lookup)
    if code_upper in GEO_PICKER:
        return GEO_PICKER[code_upper]
    
    # Fallback to full list for edge cases
    if fallback_to_all and code_upper in ALL_COUNTRIES:
        return ALL_COUNTRIES[code_upper]
    
    return code  # Return code if not found

def get_country_code(name: str) -> str | None:
    """
    Get ISO country code from human-readable name (case-insensitive).
    
    Args:
        name: Country name string (e.g., "United States", "united kingdom")
    
    Returns:
        Two-letter ISO code, or None if not found
    """
    name_lower = name.lower().strip()
    
    # Search both dictionaries
    for code, country_name in {**GEO_PICKER, **ALL_COUNTRIES}.items():
        if country_name.lower() == name_lower:
            return code
    
    return None

def is_valid_country_code(code: str) -> bool:
    """Check if a code exists in either country dictionary."""
    code_upper = code.upper().split("-")[0]
    return code_upper in GEO_PICKER or code_upper in ALL_COUNTRIES

def is_valid_time_range(backend_id: str) -> bool:
    """Check if a time range ID is valid."""
    return backend_id in TIME_RANGES

def is_valid_category(cat_id: str) -> bool:
    """Check if a category ID is valid."""
    return cat_id in CATEGORIES

def is_valid_locale(locale: str) -> bool:
    """Check if a locale string is supported."""
    return locale in LOCALES