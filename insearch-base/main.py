import pandas as pd
import plotly.express as px


def main():
    CATEGORY = 'Categoria'

    clients = pd.read_csv('resources/ClientesBanco.csv', encoding='latin1')
    clients = clients.drop(['CLIENTNUM'], axis=1)
    clients = clients.dropna()

    print(clients[CATEGORY].value_counts())
    print(clients[CATEGORY].value_counts(normalize=True))
    filter_cli = filter(lambda c: c[CATEGORY] == 'Cancelado', clients)
    print(filter_cli)

    print(clients.describe())
    print(clients.info())

    # clients.index() getLines
    for column in clients:
        fig = px.histogram(clients, x=[column], color='Categoria ')
        fig.show()


if __name__ == '__main__':
    main()
