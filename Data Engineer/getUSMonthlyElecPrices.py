
if __name__ == "__main__":
    # data engineer workload

    from API_KEY import API_KEY_EIA
    from data_list.Utilities.utils import getJsonFile, dict2csvEIA, getAPIEIA
    from data_list.Utilities.API_EIA import USMonthlyElecPricesSeriesId

    api_call = getAPIEIA(API_KEY_EIA, USMonthlyElecPricesSeriesId)

    table_dict = getJsonFile(api_call)

    df = dict2csvEIA(table_dict)
    #print(df.head())
    df.to_csv('../USMonthlyElecPrices.csv')
