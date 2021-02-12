import pandas as pd


def main():
    ID_STORE = 'ID Loja'
    FINAL_VALUE = 'Valor Final'
    QTD = 'Quantidade'
    AVERAGE = 'Ticket Medio'

    purchases = pd.read_csv('resources/vendas.csv')
    purchases[FINAL_VALUE] = purchases[FINAL_VALUE].apply(lambda value: convert_string_currency_to_decimal(value))

    group_store = purchases[[ID_STORE, FINAL_VALUE]].groupby(ID_STORE).sum()
    qtd_store = purchases[[ID_STORE, QTD]].groupby(ID_STORE).sum()

    average = (group_store[FINAL_VALUE] / qtd_store[QTD]).to_frame().rename(columns={0: AVERAGE})

    full_table = group_store.join(qtd_store).join(average)

    print(group_store.sort_values(by=FINAL_VALUE))
    print(qtd_store.sort_values(by=QTD))
    print(average)
    print(full_table)
    print(purchases[ID_STORE].unique())
    print(full_table.loc['Iguatemi Esplanada'])


def convert_string_currency_to_decimal(value: str):
    return float(value.replace(',', '.').replace('R$', ''))


if __name__ == '__main__':
    main()
