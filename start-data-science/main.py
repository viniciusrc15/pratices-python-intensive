import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os.path


# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from sklearn.model_selection import train_test_split


def generate_file():
    YEAR = 'year'
    MONTH = 'month'
    months = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10,
              'nov': 11, 'dez': 12}
    base_path = pathlib.Path('resources/dataset')
    for file in base_path.iterdir():
        name_month = file.name[:3]
        month = months[name_month]

        year = file.name[-8:]
        year = int(year.replace('.csv', ''))

        df = pd.read_csv(base_path / file.name, low_memory=False)
        df[MONTH] = month
        df[YEAR] = year
        base_airbnb = base_airbnb.append(df)

    print(list(base_airbnb.columns))
    base_airbnb.head(1000).to_csv('first_rows.csv', sep=';')


def handle_data(base_airbnb):
    final_base = base_airbnb
    for column in base_airbnb:
        if base_airbnb[column].isnull().sum() > 300000:
            final_base = base_airbnb.drop(column, axis=1)
    final_base = final_base.dropna()
    # final_base.loc[final_base['price'],final_base['new_ price']] = convert_string_currency_to_decimal(final_base['price'])
    final_base['price'] = convert_string_currency_to_decimal(final_base['price'])
    final_base['extra_people'] = convert_string_currency_to_decimal(final_base['extra_people'])
    print(final_base.dtypes)
    return final_base


def convert_string_currency_to_decimal(value):
    # TODO: fix  1.000.00
    obj = value.map(lambda a: a.replace(',', '.').replace('$', '').replace('.00', ''))
    return obj.astype(np.float32, copy=False)


def limits(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude


def exclude_outliers(df, name_column):
    qte_rows = df.shape[0]
    lim_inf, lim_sup = limits(df[name_column])
    df = df.loc[(df[name_column] >= lim_inf) & (df[name_column] <= lim_sup), :]
    rows_removed = qte_rows - df.shape[0]
    return df, rows_removed


def box_diagram(column):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=column, ax=ax1)
    ax2.set_xlim(limits(column))
    sns.boxplot(x=column, ax=ax2)


def histogram(column):
    plt.figure(figsize=(15, 5))
    sns.distplot(column, hist=True)


def graph_bar(column):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=column.value_counts().index, y=column.value_counts())
    ax.set_xlim(limits(column))


def create_diagram(base_airbnb, param: str):
    box_diagram(base_airbnb[param])
    histogram(base_airbnb[param])
    base_airbnb, linhas_removidas = exclude_outliers(base_airbnb, param)
    print('{} linhas removidas'.format(linhas_removidas))
    histogram(base_airbnb[param])
    print(base_airbnb.shape)


def create_box_diagram(base_airbnb, param):
    box_diagram(base_airbnb[param])
    graph_bar(base_airbnb[param])
    base_airbnb, linhas_removidas = exclude_outliers(base_airbnb, param)
    print('{} linhas removidas'.format(linhas_removidas))


def main():
    FILE_DISCOVER = 'resources/first_rows.csv'
    columns = ['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count',
               'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
               'beds', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee',
               'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'review_scores_cleanliness',
               'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
               'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
               'instant_bookable', 'is_business_travel_ready', 'cancellation_policy']

    if not os.path.isfile(FILE_DISCOVER):
        generate_file()

    base_airbnb = pd.read_csv(FILE_DISCOVER, error_bad_lines=False, low_memory=False, sep=';')
    base_airbnb = base_airbnb.loc[:, columns]
    base_airbnb = handle_data(base_airbnb)
    base_airbnb = base_airbnb.dropna()

    plt.figure(figsize=(15, 10))
    sns.heatmap(base_airbnb.corr(), annot=True, cmap='Greens')
    # print('-' * 60)
    # print(base_airbnb.iloc[0])

    create_diagram(base_airbnb, 'price')
    create_diagram(base_airbnb, 'extra_people')

    create_box_diagram(base_airbnb, 'host_listings_count')
    create_box_diagram(base_airbnb, 'bedrooms')
    create_box_diagram(base_airbnb, 'beds')
    create_box_diagram(base_airbnb, 'accommodates')


if __name__ == '__main__':
    main()
