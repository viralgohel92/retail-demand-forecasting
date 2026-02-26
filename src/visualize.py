import matplotlib.pyplot as plt

def plot_predictions(test, predictions, store_id=1):

    # attach predictions to dataframe
    test = test.copy()
    test['Predicted'] = predictions

    # filter one store
    store_df = test[test['Store'] == store_id]

    plt.figure(figsize=(14,6))

    plt.plot(store_df['Date'], store_df['Weekly_Sales'], label='Actual Sales')
    plt.plot(store_df['Date'], store_df['Predicted'], label='Predicted Sales')

    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.title(f"Store {store_id} - Actual vs Predicted Sales")

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()