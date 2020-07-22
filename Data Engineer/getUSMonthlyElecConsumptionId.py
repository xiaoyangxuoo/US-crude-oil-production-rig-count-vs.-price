if __name__ == "__main__":
    from API_KEY import API_KEY_EIA
    from data_list.Utilities.utils import getJsonFile, dict2csvEIA, getAPIEIA
    from data_list.Utilities.API_EIA import USMonthlyElecConsumptionId

    api_call = getAPIEIA(API_KEY_EIA, USMonthlyElecConsumptionId)

    table_dict = getJsonFile(api_call)
    df = dict2csvEIA(table_dict)

    df.to_csv('../US_monthly_elec_consump.csv')