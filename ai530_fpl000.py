import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from deeptables.models import deeptable
from tensorflow import keras


def add_Target_Value(df):
    ## Drop Duplicates that may have been occured during data collecting
    df = df.drop_duplicates(subset=["name", "element", "GW"], keep="last")

    ## Init Value change Array which is Target
    value_change = np.zeros((len(df), 1))

    for i in range(len(df)):
        # Check Gw conditions and and Same Players and get Value change from previous GWs
        if df["GW"].iloc[i] == 1:
            value_change[i] = 0
        if i - 1 < 0:
            continue
        if (df["element"].iloc[i] == df["element"].iloc[i - 1]) and (
            df["GW"].iloc[i] - df["GW"].iloc[i - 1] == 1
        ):
            value_change[i] = df["value"].iloc[i] - df["value"].iloc[i - 1]

    # print(value_change)
    print(len(value_change))

    ## Formatting train_df Dataframe to suitable form
    train_df = df.assign(valueChange=value_change)
    train_df = train_df.drop(columns=["name", "kickoff_time", "kickoff_time_formatted"])
    train_df = train_df.astype(float)
    return train_df


if __name__ == "__main__":
    df = pd.read_csv("./2016_17_merged_gw.csv", encoding="latin-1")
    # df.head()

    ### Data Preprocessing
    ## Add taget value of Value change and create an player-wise dataframe
    train_df = add_Target_Value(df.sort_values(by=["element", "GW"]))

    train_df = pd.DataFrame(train_df, dtype=object)

    # Split Tain And Test Data
    x_train, x_test, y_train, y_test = train_test_split(
        train_df.iloc[:, :-1], train_df.iloc[:, -1:], test_size=0.30, random_state=8
    )

    # Convert dataframe to Runnable form for off the shelf Models
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train, x_test, y_train, y_test = (
        x_train.to_numpy(),
        x_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )

    # print(x_train.shape, y_test.shape)

    ### Model Config and Execution
    ## Get Categorical Columns in data set
    ## Is Processed by Deep Table Nets
    cat_cols = ["element", "fixture", "id", "opponent_team", "round", "was_home"]

    ## Model Config And Init
    conf = deeptable.ModelConfig(
        nets=["dcn_nets"],
        categorical_columns=cat_cols,
        optimizer=keras.optimizers.RMSprop(),
        earlystopping_mode=False,
        earlystopping_patience=10,
    )
    dt = deeptable.DeepTable(config=conf)

    ## Model Fit
    model, history = dt.fit(x_train, y_train.flatten(), epochs=100)

    score = dt.evaluate(x_test, y_test, batch_size=1024)

    preds = dt.predict(x_test)

    print(f"Score: {score}")
    print(f"History: {history.history}")

    plt.figure(figsize=[18, 5])

    plt.subplot(1, 2, 1)
    plt.plot(
        range(len(history.history["loss"])),
        history.history["loss"],
        label="Training",
        marker=".",
    )
    plt.plot(
        range(len(history.history["val_loss"])),
        history.history["val_loss"],
        label="Testing",
        alpha=0.6,
        marker="",
    )
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        range(len(history.history["accuracy"])),
        history.history["accuracy"],
        label="Training",
        marker=".",
    )
    plt.plot(
        range(len(history.history["val_accuracy"])),
        history.history["val_accuracy"],
        label="Testing",
        alpha=0.6,
        marker="",
    )
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.legend()

    plt.show()
    plt.close()
