
def df_attr_split(df, attr, value, columns):
    train_set = df[df[attr] != value]
    test_set = df[df[attr] == value]

    train_set = train_set[columns]
    test_set = test_set[columns]
    return train_set, test_set
