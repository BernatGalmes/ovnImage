class DataFrame:

    @staticmethod
    def df_attr_split(df, attr, value, columns):
        """
        Separe a dataframe by an atribute value
        :param df: Dataframe
        :param attr: String name of the colums/attribute to split by
        :param value: Any value of attr to select
        :param columns: String[] list of columns to get of the dataframe
        :return:
        """
        train_set = df[df[attr] != value]
        test_set = df[df[attr] == value]

        train_set = train_set[columns]
        test_set = test_set[columns]
        return train_set, test_set

    @staticmethod
    def df_filter(df, filters):
        """
        Filter DataFrame Content by a list
        :param df: DataFrame witch want to filter
        :param filters: dict select the items to remove to the dataframe
                key => column, value => array of values of column of rows to discard
        :return:
        """
        df = df.copy()
        if filters is None or type(filters) is not dict:
            return df

        for key in filters:
            for value in filters[key]:
                i_delete = df.index[df[key] == value].tolist()
                df.drop(i_delete, inplace=True)
        return df
