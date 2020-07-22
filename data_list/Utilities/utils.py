"""Common API call to obtain source"""
import requests
import pandas as pd


def getJsonFile(api_call, timeout=10):
    """Return json formatted dictionary"""
    try:
        source = requests.get(api_call, timeout=timeout)
        return source.json()
    except requests.exceptions.ReadTimeout:
        print("Error loading from api call:\n"
              "%s due to server time out (current timeout limit: %s)."
              % (api_call, timeout))


def getAPIEIA(api_key, series_id):
    api_call = 'http://api.eia.gov/series/?api_key=%s&series_id=%s' \
               % (api_key, series_id)
    return api_call


def dict2csvEIA(table_dict):
    series = table_dict['series'][0]
    data = series['data']
    unit = ','.join((series['name'], series['units']))
    columns = ['YearMonth', unit]
    return pd.DataFrame(data, columns=columns)
