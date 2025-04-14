from lib import *
from transformers import standardization
from sklearn.model_selection import train_test_split

housing_path = ".\\housing_data"

def main():
    processing = True
    df = merge_files(housing_path)
    df_prepared = None
    df_labels = None
    full_pipeline = None

    train_set = None
    test_set = None

    standardized = False

    while processing:
        # df = merge_files(crypto_path)
        print(
            "Node Options\n"
            "1: DF Plot\n"
            "2: Train + Test Set Split\n" \
            "3: Strata Build\n" \
            "4: Data Explorer\n" \
            "5: Correlation Calc\n" \
            "6: Attribute Combination\n" \
            "7: Fill Missing Values\n" \
            "8: Category Encoder\n" \
            "9: Standardization Test\n" \
            "10: Linear Regression Train & Test Predict\n" \
            "11: Decision Tree Regressor\n" \
            "12: Random Forest Regressor")
        selection = int(input("Node?: "))

        match selection:
            case 1:
                plot_hist(df)
            case 2:
                train_set, test_set = split_train_test(df, 0.2)
                df_with_id = df.reset_index()
                train_set, test_set = split_train_test_by_id(df_with_id, 0.2, "index")
                print(len(train_set))
                print(len(test_set))

                # Scikit-Kit Learn provided version of above
                train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
                print(len(train_set))
                print(len(test_set))

            case 3:
                cat_name = "income_cat"
                df_col = "median_income"
                strata_build(df, cat_name, df_col, [0., 1.5, 3.0, 4.5, 6., np.inf])
            
            case 4:
                data_explorer(df, train_set)
            
            case 5:
                correlation_attrs = [
                    "median_house_value", "median_income",
                    "total_rooms", "housing_median_age"
                    ]
                calc_correlation(df, "median_house_value", correlation_attrs)

            case 6:
                attr_dict = {
                    "rooms_per_household": ["total_rooms", "households"],
                    "bedrooms_per_room": ["total_bedrooms", "total_rooms"],
                    "population_per_household": ["population", "households"]
                    }
                attr_join(df, attr_dict, "median_house_value")

            case 7:
                df = fill_missing_values(df, "total_bedrooms", 3)
            
            case 8:
                cat_processing(df, "ocean_proximity")

            case 9 :
                df_num = df.drop("ocean_proximity", axis=1)
                standardization(df_num, "median")

            case 10:
                if not standardized:
                    df_labels, df_prepared, full_pipeline = gen_prep_labels(
                        df, train_set, "median_house_value", "ocean_proximity"
                    )
                    standardized = True

                lin_reg_train(df, df_prepared, df_labels, full_pipeline)

            case 11:
                if not standardized:
                    df_labels, df_prepared, full_pipeline = gen_prep_labels(
                        df, train_set, "median_house_value", "ocean_proximity"
                    )
                    standardized = True
                    
                decision_tree_reg(df_prepared, df_labels)

            case 12:
                if not standardized:
                    df_labels, df_prepared, full_pipeline = gen_prep_labels(
                        df, train_set, "median_house_value", "ocean_proximity"
                    )
                    standardized = True

                random_forest_reg(df_prepared, df_labels)
        processing = int(input("Processing? (1 or 0) "))

if __name__ == "__main__":
    main()