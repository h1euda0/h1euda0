## import package
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor


##----------------------------------------
## Load the data
#brazil set
df_olist_customers = pd.read_csv('Data\olist_source\olist_customers_dataset.csv')
df_olist_geolocation = pd.read_csv('Data\olist_source\olist_geolocation_dataset.csv')
df_olist_order_items = pd.read_csv('Data\olist_source\olist_order_items_dataset.csv')
df_olist_order_payments = pd.read_csv('Data\olist_source\olist_order_payments_dataset.csv')
df_olist_order_reviews = pd.read_csv('Data\olist_source\olist_order_reviews_dataset.csv')
df_olist_orders = pd.read_csv('Data\olist_source\olist_orders_dataset.csv')
df_olist_products = pd.read_csv('Data\olist_source\olist_products_dataset.csv')
df_olist_sellers = pd.read_csv('Data\olist_source\olist_sellers_dataset.csv')

#support set
df_Brazil_state_region = pd.read_csv('Data\olist_source\Brazil_state_region.csv')
df_fuel_price = pd.read_csv('Data\support_file\Fuel_price_BR.csv')
df_grouped_category = pd.read_csv('Data\support_file\product_category_name_translation.csv')


##--------------------------------------------
## Data formating
##
## calculate volume
df_olist_products['volume'] = df_olist_products['product_length_cm'] * df_olist_products['product_height_cm'] * df_olist_products['product_width_cm']
##
##
## adjusted delivery date (filling NULL value)
Delivery = []
replace = df_olist_orders['order_delivered_customer_date'].isna()
for i in list(range(0,99441)):
    if replace[i] == True:
        Delivery.insert(i, df_olist_orders['order_estimated_delivery_date'][i])
    else:
        Delivery.insert(i, df_olist_orders['order_delivered_customer_date'][i])
df_olist_orders['Ajusted_delivery_date'] = Delivery
##
##
## date format
df_olist_orders['order_purchase_timestamp'] = pd.to_datetime(df_olist_orders['order_purchase_timestamp'])
df_olist_orders['Ajusted_delivery_date'] = pd.to_datetime(df_olist_orders['Ajusted_delivery_date'])
##
##
## calculate leadtime (purchase timestamp to adjusted delivery date)
leadtime = []
for i in list(range(df_olist_orders.index.argmin(), df_olist_orders.index.argmax() + 1)):
    leadtime.insert(i, (df_olist_orders['Ajusted_delivery_date'][i] - df_olist_orders['order_purchase_timestamp'][i]).days )

df_olist_orders['lead_time'] = leadtime
##
##
## date only column
df_olist_orders['date_only'] = df_olist_orders['order_purchase_timestamp'].dt.date


##----------------------------------------
## Combine Master set

## function to insert data to target table
def insert_data(column_name, connect_key, search_key, target_column, Table_name, source):
    index = list(range(Table_name.index.argmin(),Table_name.index.argmax() + 1))
    column = []

    for i in index:
        column.insert(i, source.loc[source[search_key] == Table_name[connect_key][i], target_column].iloc[0])

    Table_name[column_name] = column
    return
##
##
## add region to customers data
insert_data('region', 'customer_state', 'State_initial', 'region_initial', df_olist_customers, df_Brazil_state_region)
## add cust_region to orders
insert_data('cust_region', 'customer_id', 'customer_id', 'region', df_olist_orders, df_olist_customers )
## add cust_region to order_items
insert_data('cust_region', 'order_id', 'order_id', 'cust_region', df_olist_order_items, df_olist_orders)
## add volume to order_items
insert_data('volumne', 'product_id', 'product_id', 'volume', df_olist_order_items, df_olist_products)
## add  product category to order_items
insert_data('product_category_name', 'product_id', 'product_id', 'product_category_name', df_olist_order_items, df_olist_products)
## add region to sellers
insert_data('seller_region', 'seller_state', 'State_initial', 'region_initial', df_olist_sellers, df_Brazil_state_region )
## add sellersa_region to order_items
insert_data('seller_region', 'seller_id', 'seller_id', 'seller_region', df_olist_order_items, df_olist_sellers )
## add weight to order_items
insert_data('product_weight', 'product_id', 'product_id', 'product_weight_g', df_olist_order_items, df_olist_products)
## add leadtime to order_items
insert_data('product_weight', 'product_id', 'product_id', 'product_weight_g', df_olist_order_items, df_olist_products)
## add date_only to order_item
insert_data('order_date', 'order_id', 'order_id', 'date_only', df_olist_order_items, df_olist_orders)
## add leadtime to order_items
insert_data('lead_time', 'order_id', 'order_id', 'lead_time', df_olist_order_items, df_olist_orders)
## add grouped category
insert_data('grouped_category', 'product_category_name', 'product_category_name', 'grouped_category', df_olist_order_items, df_grouped_category)


##
##group data
##
##
## Group the data by Date, Customer region, Product type
## sum: Price, Freight_value, Volumne, weight
## max: order_item_id -> total item sell
## avg: lead_time
pre_process = df_olist_order_items.drop(columns= ['order_id','product_id', 'seller_id', 'shipping_limit_date', 'seller_region', 'product_category_name'])
result = pre_process.groupby(['order_date','cust_region', 'grouped_category']).aggregate({'volumne': 'sum', 'price': 'sum', 'freight_value':'sum', 'volumne': 'sum', 'order_item_id': 'max', 'product_weight': 'sum', 'lead_time': 'mean'})
result['total_value'] = result['price'] + result['freight_value']
result['freight_percentage'] = round((result['freight_value']/result['total_value'])*100, 2)
##
##
## add daily fuel-price
insert_data('euro95', 'order_date', 'Date', 'Euro95_BRL', result, df_fuel_price)
insert_data('diesel', 'order_date', 'Date', 'Diesel_BRL', result, df_fuel_price)
insert_data('LPG', 'order_date', 'Date', 'LPG_BRL', result, df_fuel_price)
##
##
## Combine set
df_combine = result
##

## Dummy code the data (region and product category)
df_data_dummy = pd.DataFrame()
##
##
re_dummies = pd.get_dummies(result['cust_region'], dtype= int)
category_dum = pd.get_dummies(result['grouped_category'], dtype= int)

cate_list = category_dum.columns.values.tolist()
region_list = re_dummies.columns.values.tolist()

var = ['product_weight', 'lead_time', 'price', 'euro95', 'diesel', 'LPG']
tar1 = ['total_value']
tar2 = ['volumne']
tar3 = ['value_density']

df_data_dummy[var] = result[var]

for i in list(range(result.index.argmin(), result.index.argmax() + 1)):
    df_data_dummy['year'] = result['order_date'][i].year
    df_data_dummy['month'] = result['order_date'][i].month
    df_data_dummy['day'] = result['order_date'][i].day
    
    for cate in cate_list:
        df_data_dummy[cate] = category_dum[cate][i]
        for reg in region_list:
            df_data_dummy[reg] = re_dummies[reg][i]
##
##
df_data_dummy.info()
##


##----------------------------------------------------
## Baseline model
df_unenrich = df_data_dummy.drop(columns= ['euro95', 'diesel', 'LPG'])
##
##
X  = df_unenrich
Y1 = df_combine[tar1]
Y2 = df_combine[tar2]
Y3 = df_combine[tar3]
## build Linear model
##
## split train-test
X_train_bl1, X_test_bl1, y_train_bl1, y_test_bl1 = train_test_split(X, Y1, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_bl1 = scaler.fit_transform(X_train_bl1)
X_test_bl1 = scaler.transform(X_test_bl1)
##
##
X_train_bl2, X_test_bl2, y_train_bl2, y_test_bl2 = train_test_split(X, Y2, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_bl2 = scaler.fit_transform(X_train_bl2)
X_test_bl2 = scaler.transform(X_test_bl2)
##
##
X_train_bl3, X_test_bl3, y_train_bl3, y_test_bl3 = train_test_split(X, Y3, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_bl3 = scaler.fit_transform(X_train_bl3)
X_test_bl3 = scaler.transform(X_test_bl3)
##
Baseline = LinearRegression()
##
## fit with data
Baseline.fit(X_train_bl1, y_train_bl1)
Baseline.fit(X_train_bl2, y_train_bl2)
Baseline.fit(X_train_bl3, y_train_bl3)
##
##
##
y_pred_bl1 = Baseline.predict(X_test_bl1)
mse1 = mean_squared_error(y_test_bl1, y_pred_bl1)
mse1
##
y_pred_bl2 = Baseline.predict(X_test_bl2)
mse2 = mean_squared_error(y_test_bl2, y_pred_bl2)
mse2
##
y_pred_bl3 = Baseline.predict(X_test_bl3)
mse3 = mean_squared_error(y_test_bl3, y_pred_bl3)
mse3
##
##
## Baseline hyperparameter tuning
param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'n_jobs':[10, 50, 100, 1000],
    'positive':[True, False]
}
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
gridSearch = GridSearchCV(estimator = Baseline, param_grid = param_grid, n_jobs=-1,
	cv=cvFold, scoring="neg_mean_squared_error")
##
##
searchResults_bl1 = gridSearch.fit(X_train_bl1, y_train_bl1)
searchResults_bl2 = gridSearch.fit(X_train_bl2, y_train_bl2)
searchResults_bl3 = gridSearch.fit(X_train_bl3, y_train_bl3)
##
##
best_bl1 = searchResults_bl1.best_estimator_
best_bl2 = searchResults_bl2.best_estimator_
best_bl3 = searchResults_bl3.best_estimator_
##
best_bl1.fit(X_train_bl1, y_train_bl1)
best_bl2.fit(X_train_bl2, y_train_bl2)
best_bl3.fit(X_train_bl3, y_train_bl3)
##
##
y_pred_bbl1 = best_bl1.predict(X_test_bl1)
mse1_bbl = mean_squared_error(y_test_bl1, y_pred_bbl1)
mse1_bbl
##
y_pred_bbl2 = best_bl2.predict(X_test_bl2)
mse2_bbl = mean_squared_error(y_test_bl2, y_pred_bbl2)
mse2_bbl
##
y_pred_bbl3 = Baseline.predict(X_test_bl3)
mse3_bbl = mean_squared_error(y_test_bl3, y_pred_bbl3)
mse3_bbl
###


##-------------------------------------------------------------------
## Prepare data
X  = df_data_dummy
Y1 = df_combine[tar1]
Y2 = df_combine[tar2]
Y3 = df_combine[tar3]
##
##
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, Y1, test_size=0.2, random_state=42)
##
scaler = StandardScaler()
X_train_1 = scaler.fit_transform(X_train_1)
X_test_1 = scaler.transform(X_test_1)
## y_test_3
##
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, Y2, test_size=0.2, random_state=42)
##
scaler = StandardScaler()
X_train_2 = scaler.fit_transform(X_train_2)
X_test_2 = scaler.transform(X_test_2)
##
##
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, Y3, test_size=0.2, random_state=42)
##
scaler = StandardScaler()
X_train_3 = scaler.fit_transform(X_train_3)
X_test_3 = scaler.transform(X_test_3)
##
##
## SVR buiuld
model_SVR = SVR()
##
## param
kernel = ["linear","rbf", "sigmoid", "poly"]
tolerance = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
C = np.arange(1 ,10 , 0.5)
grid = dict(kernel=kernel, tol=tolerance, C=C)
## gridsearch
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
gridSearch = GridSearchCV(estimator=model_SVR, param_grid=grid, n_jobs=-1,
	cv=cvFold, scoring="neg_mean_squared_error")

## search
searchResults_svr1 = gridSearch.fit(X_train_1, y_train_1)
searchResults_svr2 = gridSearch.fit(X_train_2, y_train_2)
searchResults_svr3 = gridSearch.fit(X_train_3, y_train_3)

##
best_svr1 = searchResults_svr1.best_estimator_
best_svr2 = searchResults_svr2.best_estimator_
best_svr3 = searchResults_svr3.best_estimator_
##
##
best_svr1.fit(X_train_1, y_train_1)
y_pred_svr1 = best_svr1.predict(X_test_1)
mse2_svr1 = mean_squared_error(y_test_1, y_pred_svr1)
mse2_svr1
##
best_svr2.fit(X_train_2, y_train_2)
y_pred_svr2 = best_svr2.predict(X_test_2)
mse2_svr2 = mean_squared_error(y_test_2, y_pred_svr2)
mse2_svr2
##
best_svr3.fit(X_train_3, y_train_3)
y_pred_svr3 = best_svr3.predict(X_test_3)
mse2_svr3 = mean_squared_error(y_test_3, y_pred_svr3)
mse2_svr3

##
## CART/DT regressor
##
DTR = DecisionTreeRegressor()
## param
parameters={"splitter":["best","random"],
            "max_depth" : [7,9,11,12],
           "min_samples_leaf":[5,6,7,8,9,10,11,12,13],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,40,50,60,70,80,90,100,120,150] }
## grid search
tuning_model=GridSearchCV(DTR,param_grid=parameters,scoring='neg_mean_squared_error',cv=3,verbose=3)

##
searchresult_DTR1 = tuning_model.fit(X_train_1, y_train_1)
searchresult_DTR2 = tuning_model.fit(X_train_2, y_train_2)
searchresult_DTR3 = tuning_model.fit(X_train_3, y_train_3)

##
best_param_DT1 = searchresult_DTR1.best_estimator_
best_param_DT2 = searchresult_DTR2.best_estimator_
best_param_DT3 = searchresult_DTR3.best_estimator_

##
best_DTR1 = DecisionTreeRegressor( splitter= best_param_DT1['splitter'],
                                  max_depth= best_param_DT1['max_depth'],
                                  min_samples_leaf= best_param_DT1['min_samples_leaf'],
                                  min_weight_fraction_leaf= best_param_DT1['min_weight_fraction_leaf'],
                                  max_features= best_param_DT1['max_features'],
                                  max_leaf_nodes= best_param_DT1['min_samples_leaf'])
##
best_DTR2 = DecisionTreeRegressor( splitter= best_param_DT2['splitter'],
                                  max_depth= best_param_DT2['max_depth'],
                                  min_samples_leaf= best_param_DT2['min_samples_leaf'],
                                  min_weight_fraction_leaf= best_param_DT2['min_weight_fraction_leaf'],
                                  max_features= best_param_DT2['max_features'],
                                  max_leaf_nodes= best_param_DT2['min_samples_leaf'])
##
best_DTR3 = DecisionTreeRegressor( splitter= best_param_DT3['splitter'],
                                  max_depth= best_param_DT3['max_depth'],
                                  min_samples_leaf= best_param_DT3['min_samples_leaf'],
                                  min_weight_fraction_leaf= best_param_DT3['min_weight_fraction_leaf'],
                                  max_features= best_param_DT3['max_features'],
                                  max_leaf_nodes= best_param_DT3['min_samples_leaf'])

##
best_DTR1.fit(X_train_1, y_train_1)
y_pred_dtr1 = best_DTR1.predict(X_test_1)
mse_dtr1 = mean_squared_error(y_test_1, y_pred_dtr1)
mse_dtr1
##
best_DTR2.fit(X_train_2, y_train_2)
y_pred_dtr2 = best_DTR1.predict(X_test_2)
mse_dtr2 = mean_squared_error(y_test_2, y_pred_dtr2)
mse_dtr2
##
best_DTR3.fit(X_train_3, y_train_3)
y_pred_dtr3 = best_DTR3.predict(X_test_3)
mse_dtr3 = mean_squared_error(y_test_3, y_pred_dtr3)
mse_dtr3
##