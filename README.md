# FFOOD
**Framework for Feature and Observation Outlier Detection (FFOOD)**

The package is nice and simple and boils down to one command.

```python
outliers, features = tables(clean_data)
```

#### Description

FFOOD is a unique method to audit potential model inputs. It is designed to help you identify whether you might need additional variables and whether you have mistakes or outliers in your datasets.

FFOOD addresses whether confounders or outliers drive the prediction error. This method is primarily meant for cross-sectional datasets. All functions only work with machine readible data; do no pass it NaN or Infinite values and make sure that all category variables are one-hot encoded.

The method is at first an analysis of the most overpredicted instances (hence most underpredicted too). The method starts by dividing the dataset in two random subsets. The model is trained on subset one to predict subset two, after which it is trained on subset two to predict subset one. The feature to be predicted is one plus log transformed to ensure there is no negative predictions.

The percentage difference between the predicted and the actual target is used to determine the most overpredicted and under predicted instances. This process is repeated N number of times different test-train samples to ensure that a stable overpredicted and underpredicted percentage have been obtained.

The next step is to find the features most associated with the overprediction. At this point one knows which instances are predicted outliers and what features are driving this outliers. While paying close attention to both the outlier instances and outlier features the following three questions have to be asked.


- Going back to the unstructured data, is there any additional features (confounders) that you could have missed?
- If no additional attributes or characteristics coul help to explain the overprediction, is the instance a data entry mistake or an outlier?
- Have a look at the unsupervised feature characteristics like predictability, informativeness, underpredictor, overpredictor and, outlier-driver and repeat the first two steps.

The best way to dealt with this issue is with an example. This example uses an Airbnb dataset. 

# Example

#### Airbnb Daily Fair Valuation


Welcome to Airbnb Analysis Corp.! Your task is to set the competitive ****daily accomodation rate**** for a client's house in Bondi Beach. The owner currently charges $500. We have been tasked to estimate a ****fair value**** that the owner should be charging. The house has the following characteristics and constraints. While developing this model you came to realise that Airbnb can use your model to estimate the fair value of any property on their database, your are effectively creating a recommendation model for all prospective hosts!

1. The owner has been a host since ****August 2010****
1. The location is ****lon:151.274506, lat:33.889087****
1. The current review score rating ****95.0****
1. Number of reviews ****53****
1. Minimum nights ****4****
1. The house can accomodate ****10**** people.
1. The owner currently charges a cleaning fee of ****370****
1. The house has ****3 bathrooms, 5 bedrooms, 7 beds****.
1. The house is available for ****255 of the next 365 days****
1. The client is ****verified****, and they are a ****superhost****.
1. The cancelation policy is ****strict with a 14 days grace period****.
1. The host requires a security deposit of ****$1,500****

```python
    from dateutil import parser
    dict_client = {}
    dict_client["city"] = "Bondi Beach"
    dict_client["longitude"] = 151.274506
    dict_client["latitude"] = -33.889087
    dict_client["review_scores_rating"] = 95
    dict_client["number_of_reviews"] = 53
    dict_client["minimum_nights"] = 4
    dict_client["accommodates"] = 10
    dict_client["bathrooms"] = 3
    dict_client["bedrooms"] = 5
    dict_client["beds"] = 7
    dict_client["security_deposit"] = 1500
    dict_client["cleaning_fee"] = 370
    dict_client["property_type"] = "House"
    dict_client["room_type"] = "Entire home/apt"
    dict_client["availability_365"] = 255
    dict_client["host_identity_verified"] = 1  ## 1 for yes, 0 for no
    dict_client["host_is_superhost"] = 1
    dict_client["cancellation_policy"] = "strict_14_with_grace_period"
    dict_client["host_since"] = parser.parse("01-08-2010")
```
<br />

**Raw Data**

```python
raw_data = pd.read_csv("https://github.com/firmai/random-assets/blob/master/listings.csv?raw=true")
```

<br />

**Cleaned Data**

```python
clean_data = your_cleaning_operations(raw_data)
```

<br />

### Start Here

<br />

**FFOOD Tables**

```python
outliers, features = tables(clean_data)
````

<br />

**Outliers**

This operation finds the prediction outlier for all feature. The first is an anlysis of 'price' as the target. 

```python
outliers[outliers["Predicted Feature"]=="price"]
```

| Overprediction Index | Overpredict Percentage | Underprediction Index | Underpredict Percentage | Predicted Feature | Top Feature          | ABS SHAP Value | Larger Feature Leads to Overprediction (FLO) | FLO Value  | Larger Feature Leads to Underprediction (FLU) | FLU Value  |
| -------------------- | ---------------------- | --------------------- | ----------------------- | ----------------- | -------------------- | -------------- | -------------------------------------------- | ---------- | --------------------------------------------- | ---------- |
| 18039                | 900                    | 33057                 | -96                     | price             | Entire home/apt      | 204941.78      | Private room                                 | 5678.24536 | bathrooms                                     | 2219.96877 |
| 32657                | 629                    | 30218                 | -95                     | price             | accommodates         | 184487.462     | security_deposit                             | 903.426214 | Entire home/apt                               | 846.470856 |
| 18416                | 441                    | 24287                 | -94                     | price             | bathrooms            | 164815.563     | longitude                                    | 237.58082  | bathrooms_per_person                          | 503.535783 |
| 32731                | 431                    | 27492                 | -93                     | price             | bedrooms             | 122461.56      | cleaning_fee                                 | 191.342111 | accommodates                                  | 445.722047 |
| 27078                | 351                    | 30957                 | -93                     | price             | bathrooms_per_person | 96394.2702     | beds                                         | 105.192666 | Shared room                                   | 394.079955 |

<br />

The next is the same table but for the average reviewer rating. All feature are are contained within the *outliers* data frame.

```python
outliers[outliers["Predicted Feature"]=="review_scores_rating"]
```

| Overprediction Index | Overpredict Percentage | Underprediction Index | Underpredict Percentage | Predicted Feature    | Top Feature                | ABS SHAP Value | Larger Feature Leads to Overprediction (FLO) | FLO Value  | Larger Feature Leads to Underprediction (FLU) | FLU Value  |
| -------------------- | ---------------------- | --------------------- | ----------------------- | -------------------- | -------------------------- | -------------- | -------------------------------------------- | ---------- | --------------------------------------------- | ---------- |
| 15082                | 393                    | 32944                 | -23                     | review_scores_rating | number_of_reviews          | 17494.1253     | bedrooms                                     | 205.592071 | Private room                                  | 304.302714 |
| 24905                | 384                    | 32943                 | -20                     | review_scores_rating | past_and_future_popularity | 13891.5491     | security_deposit                             | 194.578755 | accommodates                                  | 124.632332 |
| 29090                | 380                    | 31644                 | -20                     | review_scores_rating | latitude                   | 2382.14212     | host_identity_verified                       | 95.738997  | number_of_reviews                             | 68.089677  |
| 5375                 | 377                    | 16751                 | -20                     | review_scores_rating | accommodates               | 2242.57177     | latitude                                     | 62.409664  | longitude                                     | 62.943133  |
| 3210                 | 374                    | 31149                 | -19                     | review_scores_rating | host_is_superhost          | 1933.74017     | host_is_superhost                            | 13.874651  | bathrooms_per_person                          | 39.594302  |



*From here forward, I will focus on the price feature as target.*

<br />

**Raw Data**


| Overprediction Instances                                                       | Archive                                                  | Underprediction Instances                                                      | Archive                                                  |
| ------------------------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------- |
| [https://www.airbnb.com/rooms/21743681](https://www.airbnb.com/rooms/21743681) | [http://archive.today/BR1ss](http://archive.today/BR1ss) | [https://www.airbnb.com/rooms/26932284](https://www.airbnb.com/rooms/26932284) | [http://archive.today/i2AjM](http://archive.today/i2AjM) |
| [https://www.airbnb.com/rooms/21884828](https://www.airbnb.com/rooms/21884828) | [http://archive.today/dIyVM](http://archive.today/dIyVM) | [https://www.airbnb.com/rooms/30043604](https://www.airbnb.com/rooms/30043604) | [http://archive.today/ttcqI](http://archive.today/ttcqI) |
| [https://www.airbnb.com/rooms/29807040](https://www.airbnb.com/rooms/29807040) | [http://archive.today/3I9GP](http://archive.today/3I9GP) | [https://www.airbnb.com/rooms/31601306](https://www.airbnb.com/rooms/31601306) | [http://archive.today/uc8m3](http://archive.today/uc8m3) |
| [https://www.airbnb.com/rooms/33861409](https://www.airbnb.com/rooms/33861409) | [http://archive.today/EPLdO](http://archive.today/EPLdO) | [https://www.airbnb.com/rooms/32384612](https://www.airbnb.com/rooms/32384612) | [http://archive.today/EDKtZ](http://archive.today/EDKtZ) |
| [https://www.airbnb.com/rooms/33912597](https://www.airbnb.com/rooms/33912597) | [http://archive.today/IeHQ9](http://archive.today/IeHQ9) | [https://www.airbnb.com/rooms/34231022](https://www.airbnb.com/rooms/34231022) | [http://archive.today/SclC0](http://archive.today/SclC0) |





There is a lot of other raw data that can be found here for the [overpredicted](https://github.com/firmai/FFOOD/blob/master/raw/Over.csv) and [underpredicted](https://github.com/firmai/FFOOD/blob/master/raw/Under.csv) instances. 

</br >

**Features**

The figures to this table rely on the entire data set and are not specific to any one feature.

| predictability Feature     | predictability Value | informativeness Feature    | informativeness Value | overpredictor Feature | overpredictor Value | underpredictor Feature     | underpredictor Value | outlier_driver Feature | outlier_driver Value |
| -------------------------- | -------------------- | -------------------------- | --------------------- | --------------------- | ------------------- | -------------------------- | -------------------- | ---------------------- | -------------------- |
| number_of_reviews          | 236616.671           | availability_365           | 249729.3              | Private room          | 1742.98441          | accommodates               | 3961.16197           | availability_365       | 16876.2              |
| price                      | 154620.127           | past_and_future_popularity | 149925.182            | bedrooms              | 1359.6978           | past_and_future_popularity | 1847.40775           | number_of_reviews      | 1414.2               |
| security_deposit           | 116134.893           | Entire home/apt            | 82007.025             | bathrooms_per_person  | 976.245801          | host_is_superhost          | 922.621391           | minimum_nights         | 842.2                |
| past_and_future_popularity | 43272.0684           | cleaning_fee               | 64619.9799            | bedrooms_per_person   | 500.608543          | bathrooms_per_person       | 904.433052           | price                  | 644.6                |
| cleaning_fee               | 30418.8392           | bathrooms                  | 55519.1308            | Hotel room            | 338.967136          | bathrooms                  | 824.314968           | beds                   | 475.6                |



