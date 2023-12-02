## import package
import pandas as pd


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


##
##group data
##
##
## Group the data by Date, Customer region, Product type
## sum: Price, Freight_value, Volumne, weight
## max: order_item_id -> total item sell
## avg: lead_time
pre_process = df_olist_order_items.drop(columns= ['order_id','product_id', 'seller_id', 'shipping_limit_date', 'seller_region'])
result = pre_process.groupby(['order_date','cust_region', 'product_category_name']).aggregate({'volumne': 'sum', 'price': 'sum', 'freight_value':'sum', 'volumne': 'sum', 'order_item_id': 'max', 'product_weight': 'sum', 'lead_time': 'mean'})
result['total_value'] = result['price'] + result['freight_value']
result['freight_percentage'] = round((result['freight_value']/result['total_value'])*100, 2)